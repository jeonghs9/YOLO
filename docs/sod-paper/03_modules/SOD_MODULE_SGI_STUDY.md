# SGI 공부 노트 — Semantic-Guided Injection (코드 1:1)

> 학습용 정리. 실제 코드(`2-fpn`: `ultralytics/nn/modules/block.py:SGI`, config `yolo26-p2-sgiV3.yaml`)와 1:1.
> 설계 배경/검증은 `../03_modules/SOD_MODULE_SGI.md`, 진단은 `../04_results/SOD_RESULT_AP_SIZE.md`.

---

## 1. 이 모듈의 역할 (한 줄)

**고해상이지만 의미가 빈약한 P2 검출층에, 깊은 층(P3/P4/P5)의 "의미"를 선택적으로 주입해 작은 물체의 검출(recall→AP)을 끌어올린다.**

> **초급자 설명:** P2는 "시력 좋은데 글 못 읽는 학생", 깊은 층은 "글 잘 읽는 선배". SGI는 선배의 이해를 학생에게 **중요한 내용만, 중요한 자리에** 전달해 학생이 작은 글자(작은 물체)를 알아보게 합니다.

---

## 2. 어디에 있나 (위치)

YOLO26-p2 head의 **layer 20** (P2 검출 직전). base = P2 feature(layer 19), 소스 = P3td(16)·P4td(13)·P5(10).

```
[Backbone]                                   stride
 2 C3k2  ── P2 feature ───────────────────────  4  ──┐(base)
 4 C3k2  ── P3 feature                            8   │
 6 C3k2  ── P4 feature                           16   │
10 C2PSA ── P5 feature                           32   │
[Head: top-down]                                      │
11 Up→12 Concat(P4=6)→13 C3k2 = P4-td (16) ──┐(src)   │
14 Up→15 Concat(P3=4)→16 C3k2 = P3-td  (8) ──┤(src)   │
17 Up→18 Concat(P2=2)→19 C3k2 = P2 base (4) ─┼────────┘
                                             │
★20 SGI(base=19, src={16,13,10})  →  의미 주입된 P2 (stride 4)
[Head: bottom-up]
21↓→22 Concat(16)→23 C3k2 = P3 out
24↓→25 Concat(13)→26 C3k2 = P4 out
27↓→28 Concat(10)→29 C3k2 = P5 out
[Detect] 30: [20(P2), 23(P3), 26(P4), 29(P5)]
```
> SGI는 P2 검출층 하나만 강화. P3/P4/P5 경로·검출은 그대로. (V1=src{16}, V2={16,13}, V3={16,13,10})

---

## 3. 모듈 내부 데이터 흐름

```
 x0 (P2 base, L19) ──[ base_proj = Identity ]────────────────────────────► B ──────────────┐
                                                                                            │
 src x_i (L16,13,10) ─[Conv1x1_i]─[Up→P2크기]─[× w_i]──(Σ)──► F_sem ─┬─[GAP→1x1→ReLU→1x1→σ]─► a_c (C×1×1)
   (P3td,P4td,P5)       src_proj    nearest    w=softmax(θ)         │                         │
                                                                    │            (a_c ⊙ F_sem)│
                                                                    ├─[mean_c;max_c]→Conv7x7→σ─► a_s (1×H×W)
                                                                    │                         │
                                                                    └──► inj = a_s ⊙ (a_c ⊙ F_sem)
                                                                                            │
                              out = B + γ · inj  ◄────────────────────────────────────────────┘
```
- 채널 attention `a_c` = "어떤 정보 채널(what)", 공간 attention `a_s` = "어느 위치(where)".
- `γ`(init 0): 처음엔 `out = B`(주입 0) → 학습으로 세기·부호 조절.

---

## 4. 수식 (LaTeX)

입력: base $x_0$, 소스 $\{x_1,\dots,x_n\}$. $\mathrm{Up}$ = base 크기로 nearest 업샘플, $\sigma$ = sigmoid, $\odot$ = 브로드캐스트 원소곱, $\mathrm{GAP}$ = global average pool, $\mu_c/\max_c$ = 채널축 평균/최대.

$$
\mathbf{B} =
\begin{cases}
x_0 & \text{if } c_{0}=c_2 \;(\text{Identity})\\
\mathrm{Conv}_{1\times1}(x_0) & \text{otherwise}
\end{cases}
$$

$$
\mathbf{w}=\mathrm{softmax}(\boldsymbol{\theta}),\quad \boldsymbol{\theta}\in\mathbb{R}^{n}\ (\text{init }\mathbf{0})
$$

$$
\mathbf{F}_{\mathrm{sem}}=
\begin{cases}
\mathbf{B} & n=0 \quad(\text{V0: self})\\[2pt]
\displaystyle\sum_{i=1}^{n} w_i\,\mathrm{Up}\!\big(\mathrm{Conv}^{(i)}_{1\times1}(x_i)\big) & n\ge 1
\end{cases}
$$

$$
\mathbf{a}_c=\sigma\!\Big(\mathbf{W}_2\,\mathrm{ReLU}\big(\mathbf{W}_1\,\mathrm{GAP}(\mathbf{F}_{\mathrm{sem}})\big)\Big)\in\mathbb{R}^{C\times1\times1}
$$

$$
\mathbf{a}_s=\sigma\!\Big(\mathrm{Conv}_{7\times7}\big[\,\mu_c(\mathbf{F}_{\mathrm{sem}})\,;\,\max_c(\mathbf{F}_{\mathrm{sem}})\,\big]\Big)\in\mathbb{R}^{1\times H\times W}
$$

$$
\mathbf{inj}=
\begin{cases}
\mathbf{a}_s\odot(\mathbf{a}_c\odot\mathbf{F}_{\mathrm{sem}}) & \text{use\_attn}\\
\mathbf{F}_{\mathrm{sem}} & \text{otherwise (control, V3noattn)}
\end{cases}
$$

$$
\boxed{\;\mathbf{out}=\mathbf{B}+\gamma\,\mathbf{inj}\;},\qquad \gamma\in\mathbb{R}\ (\text{init }0,\ \text{unconstrained})
$$

**핵심 성질:** 채널 일치 시 $\mathbf{B}=x_0$이고 $\gamma=0$이면 $\mathbf{out}=x_0$ → **시작은 정확히 naive-P2**(순수 residual add-on). V0에서 $\mathbf{F}_{\mathrm{sem}}=\mathbf{B}$라 self-attention.

---

## 5. 수식 ↔ 코드 1:1 매핑

| 수식 | block.py: SGI 코드 |
|---|---|
| $\mathbf{B}=\mathrm{Identity}(x_0)$ or $\mathrm{Conv}(x_0)$ | `self.base_proj = nn.Identity() if c1[0]==c2 else Conv(c1[0],c2,1)` / `base = self.base_proj(x[0])` |
| $\mathbf{w}=\mathrm{softmax}(\boldsymbol\theta)$, $\theta$ init 0 | `self.src_w = nn.Parameter(torch.zeros(self.n_src))` / `w_src = torch.softmax(self.src_w, dim=0)` |
| $\mathbf{F}_{sem}=\sum_i w_i\mathrm{Up}(\mathrm{Conv}^{(i)}(x_i))$ | `for i,proj in enumerate(self.src_proj): s=proj(x[i+1]); s=F.interpolate(s,size=(h,w),"nearest"); term=w_src[i]*s; f_sem += term` |
| $\mathbf{F}_{sem}=\mathbf{B}$ (V0) | `if self.n_src==0: f_sem = base` |
| $\mathbf{a}_c=\sigma(W_2\mathrm{ReLU}(W_1\mathrm{GAP}(\cdot)))$ | `self.ca = Seq(AdaptiveAvgPool2d(1), Conv2d(c2,hidden,1), ReLU, Conv2d(hidden,c2,1), Sigmoid)` / `a_c=self.ca(f_sem)` |
| $\mathbf{a}_s=\sigma(\mathrm{Conv}_{7\times7}[\mu_c;\max_c])$ | `sa_in=cat([f_sem.mean(1,keepdim),f_sem.amax(1,keepdim)],1)` / `a_s=self.sa(sa_in)` (`sa=Seq(Conv2d(2,1,7,pad3),Sigmoid)`) |
| $\mathbf{inj}=\mathbf{a}_s\odot(\mathbf{a}_c\odot\mathbf{F}_{sem})$ | `inj = a_s * (a_c * f_sem)` |
| $\mathbf{inj}=\mathbf{F}_{sem}$ (control) | `else: inj = f_sem` |
| $\mathbf{out}=\mathbf{B}+\gamma\,\mathbf{inj}$ | `return base + self.gamma * inj` (`self.gamma=Parameter(torch.zeros(1))`) |

(검증: 수식대로 독립 재계산 → 모듈 출력과 `allclose=True`. `../03_modules/SOD_MODULE_SGI.md` §4.)

---

## 6. 어떻게 작동하나 (단계별)

1. **base 통과** — P2 feature를 그대로(Identity) 들고 감 → 고해상 detail 보존.
2. **소스 정렬·집계** — P3/P4/P5를 1×1로 채널 맞추고 P2 크기로 키운 뒤, 학습된 비중 $w$로 가중합 → 의미 feature $F_{sem}$.
3. **무엇(채널)** — $F_{sem}$에서 의미 있는 채널을 $a_c$로 강조.
4. **어디(위치)** — 물체 있을 법한 위치를 $a_s$로 강조.
5. **주입** — $a_c,a_s$로 걸러진 의미를 $\gamma$만큼 base에 더함.

> **초급자 설명:** ① 학생 노트(P2)는 그대로 두고 → ② 선배 3명 설명을 비중 조절해 한 장으로 합치고 → ③ 그중 핵심 내용만(채널) → ④ 중요한 줄에만(위치) → ⑤ 학생 노트에 살짝 덧붙입니다. 덧붙이는 양(γ)은 학습으로 정해요.

---

## 7. 기존 연구와의 차이 · 창의성 · 아이디어 출처

### 구성 요소는 익숙함 (정직하게)
1×1 projection, SE식 채널 attention, CBAM식 공간 attention, BiFPN식 가중 융합, residual gate(γ) — 각각은 알려진 블록. **"새 블록"만으로는 incremental.**

### 차별점 (재조합 + 타깃)
| 기존 | 방식 | SGI |
|---|---|---|
| PAN(YOLO 내장) | 얕은+깊은 단순 concat | 단순결합 X — 의미 소스가 주입 위치·채널을 **guide** |
| BiFPN | 입력별 **스칼라** 가중 | 채널·**위치별(per-pixel)** content-aware |
| AugFPN | **최상위** 보강 | **최하위 고해상(P2, tiny가 사는 곳)** 보강 |
| CBAM | self-attention | **cross-level**(깊은층→P2). 그 대조군이 V0 |
| CARAFE | 업샘플만 | residual 주입으로 detail 보존 결합 |

→ **창의성 = "깊은 의미가 고해상 층의 어디·무엇에 주입될지 스스로 정하고(semantic-guided), 원본 detail은 residual로 보존하며, 소스 깊이까지 학습 선택"** 을 **소형 객체 전용**으로 묶고 **size별 AP/AR + 대조군(V0/V3noattn)**으로 검증한 점.

### 아이디어는 어디서 왔나 (= 우리 데이터)
**빌려온 게 아니라 진단에서 도출:**
1. **loss 8종 실패** → APtiny 노이즈 초과 0개 → "병목은 loss가 아니라 구조".
2. **naive-P2**: 전반 AP +0.0195인데 tiny는 **recall↑ / AP→** → "P2는 해상도는 주는데 의미가 약하다(semantic gap)".
3. **matched-IoU 진단**: tiny 실패는 **검출(의미) 1차** + 정밀도(기하) 2차 → "**고해상에 의미를 주입**하자"가 직접 도출.

→ 메커니즘 선택(attention 형태)은 CBAM/SE/BiFPN에서 효율적 블록으로 차용, **무엇을 풀지(고해상+의미)는 우리 데이터가 지정**.

> **초급자 설명:** "남이 만든 멋진 모듈을 가져온" 게 아니라, **우리 실험이 "작은 물체는 고해상에 의미가 없어서 못 잡는다"를 보여줘서**, 거기에 딱 맞게 "의미를 고해상에 넣는" 모듈을 우리 손으로 조립한 거예요. 부품은 흔하지만 **왜·어디에 쓰는지가 우리 발견**입니다.

### 정직한 한계 (검토 반영)
- "완전히 새 구조"는 아님 — **SOD 목적에 재배치한 경량 cross-level attention injection**. novelty는 수식보다 **진단→설계→size별 검증의 문제해결 스토리 + 대조군**에 있음.
- 효과는 **학습으로만 확정**(현재 V0~V3 학습 중). γ가 0 근처면 미학습 의심 → `inspect_sgi.py`로 점검.

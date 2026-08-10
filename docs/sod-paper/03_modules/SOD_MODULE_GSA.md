# GSA — Guided Sharp Attention (SGI 실패 후속 모듈)

브랜치: `2.2-fpn_gsa` (← `2-fpn`). 코드: `ultralytics/nn/modules/block.py: GSA` / config: `yolo26-p2-gsa.yaml`.

> ## ⚠️ 최종 판정 (2026-07-02): negative — GSA도 무효
> RUN_260701_01 (pycocotools test): **tiny 0.0631 = naive-P2와 정확히 동일**, 전체 AP 0.1851(naive-P2 0.1841 노이즈 내; SGI-V3 0.1882보다 낮음). **학습된 γ = −0.91(음수)** → 모델이 GSA 출력을 **빼는 방향**으로 학습(=순효과 부정). src_w=[P3 0.45, P4 0.30, P5 0.25].
> → **"흐리기 전에 선명하게 옮기기"도 tiny를 못 올림.** SGI(주입)에 이어 GSA(선명 주입)도 실패 → **"깊은 의미를 P2로 가져오는 모듈" 계열은 대조군 포함 종결.** 갭(의미)은 top-down이 이미 메웠고, 선명히 옮겨도 중복이라 무효.
> → 다음 레버는 P2 fusion이 아니라 **assignment·해상도·기하**. 상세: `../04_results/SOD_RESULT_AP_SIZE.md`.
> 아래 설계·검증 기록은 보존(코드 정확성 유효, 논문 대조 증거).

> ## ⚠️⚠️ 코덱스 정정 (2026-07-02): "흐리기 전에 선명하게"는 코드와 불일치 — 문서/논문에서 삭제
> **실제 코드 흐름:** `deep source → 1×1 conv → nearest 업샘플(→P2 해상도) → q/k/v → 3×3 unfold local attention`.
> 즉 attention은 **이미 nearest로 흐려진(coarse) deep map 위에서** 수행된다. **"흐리기 전(before blur)"이 아니다.**
> GSA가 한 일 = **coarse semantic prior를 P2 위에서 content-adaptive하게 재혼합**한 것 (native deep grid 참조가 아님).
> 진짜 "흐리기 전"이려면 high-res query가 **deep native grid**를 직접 참조하거나 **deformable/cross-scale sampling**으로 원 좌표계에서 가져와야 함 → 현재 GSA는 그게 아님.
> **아래 본문의 "흐리기 전에/선명하게 복원" 표현은 모두 이 정정에 종속** — 실패 원인 설명과도 일치(coarse prior 재혼합이라 새 detail 없음, γ<0).

---

> **왜 GSA를 시도했나:** SGI(의미 주입)는 negative였다(`../03_modules/SOD_MODULE_SGI.md`) — 이유는 ① 의미는 top-down이 이미 넣음 ② SGI가 **nearest 업샘플(흐림) 후 강조**라 detail이 이미 손실. GSA는 ②를 노리고 **고해상이 깊은 층에 "질문"해 자리별로 골라오는** local attention을 붙였다. (단, 위 정정대로 GSA도 nearest 업샘플 뒤에서 attention하므로 native detail 복원은 아님.)

---

## 1. 한 줄 컨셉 (⚠️ 위 정정 반영)

**의도:** 선명한 P2가 "여기 뭐가 있어?"라고 질문하면 깊은 층이 자리별로 골라 건넨다.
**실제:** nearest로 업샘플된 coarse deep map 위에서 3×3 local attention으로 content-adaptive 재혼합. (native detail 새로 생성 X)

> **초급자 설명:** 원래 의도는 "늘리는 순간 또렷하게 채우기"였는데, **코드는 이미 흐리게 늘린 뒤 그 위에서 고르는** 방식이었어요. 그래서 "흐린 걸 다시 섞은" 셈이라 새 정보가 안 생겼고, 결과도 안 좋았습니다.

---

## 2. 그림

```
 깊은 층 F_d (똑똑/흐림)                선명한 층 base=P2 (또렷/얕음) = 질문자
    │  각 자리 주변 3×3 창 d₁…d₉           │  q = 이 자리의 질문
    ▼ (열쇠 k, 값 v)                      ▼
 [ k₁…k₉ , v₁…v₉ ] ───► α = softmax(q·k/√c)  ← "어느 후보가 중요?"(맥락)
                              │
        선명한 의미 pull = Σ αₘ·vₘ  ◄──┘
                              │
     out = base + γ·pull   ◄── 원본은 그대로, γ=0에서 시작(안전)
```

---

## 3. 수식 (LaTeX)

고해상 위치 $x$에서, 깊은 층의 $k\times k$ 창 $\{d_m\}_{m=1}^{K}$ ($K=k^2$, 기본 $k=3$):

$$
\mathbf{base}=\begin{cases}x_0 & c_0=c_2\ (\text{Identity})\\ \mathrm{Conv}_{1\times1}(x_0) & \text{else}\end{cases}
\qquad
\mathbf{F}_d=\sum_i \mathrm{softmax}(\boldsymbol\theta)_i\,\mathrm{Up}\!\big(\mathrm{Conv}^{(i)}_{1\times1}(x_i)\big)
$$

$$
q=\mathrm{Conv}_q(\mathbf{base}),\quad k_m=\mathrm{Conv}_k(\mathbf{F}_d)_m,\quad v_m=\mathrm{Conv}_v(\mathbf{F}_d)_m
$$

$$
\alpha_m=\frac{\exp\!\big(q^\top k_m/\sqrt{c}\big)}{\sum_{m'}\exp\!\big(q^\top k_{m'}/\sqrt{c}\big)}\quad(\textstyle\sum_m\alpha_m=1)
$$

$$
\mathrm{pull}(x)=\sum_{m=1}^{K}\alpha_m\,v_m
\qquad
\boxed{\,\mathbf{out}(x)=\mathbf{base}(x)+\gamma\cdot\mathrm{pull}(x)\,},\quad\gamma\ \text{init }0
$$

- $q$=선명한 층의 질문, $k_m/v_m$=깊은 층 후보의 열쇠/값, $\alpha_m$=중요도(맥락 이해), $\gamma$=주입 게이트(0에서 시작, 부호도 학습).
- 채널 일치 시 base=Identity → **$\gamma=0$이면 out=원본 P2**(순수 residual, 안전초기화).

---

## 4. 수식 ↔ 코드 1:1

| 수식 | block.py: GSA |
|---|---|
| base (Identity/Conv) | `self.base_proj = nn.Identity() if c1[0]==c2 else Conv(c1[0],c2,1)` / `base=self.base_proj(x[0])` |
| $F_d=\sum \mathrm{softmax}(\theta)_i\mathrm{Up}(\mathrm{Conv}_i(x_i))$ | `w_src=softmax(src_w)` + 루프 `s=proj(x[i+1]); F.interpolate(...,"nearest"); f_d += w_src[i]*s` |
| $q,k_m,v_m$ | `q=self.q(base); key=self.k_proj(f_d); val=self.v(f_d)` |
| $k×k$ 창 | `F.unfold(key,k,padding=pad).view(b,c,K,h,w)` (val 동일) |
| $\alpha=\mathrm{softmax}(q\cdot k/\sqrt c)$ | `attn=(q.unsqueeze(2)*k_unf).sum(1)*scale; attn=attn.softmax(1)` |
| $\mathrm{pull}=\sum\alpha_m v_m$ | `pull=(attn.unsqueeze(1)*v_unf).sum(2)` |
| $\mathrm{out}=base+\gamma\,pull$ | `return base + self.gamma*pull` |

---

## 5. SGI와의 차이 (핵심 novelty)

| | SGI (negative) | GSA |
|---|---|---|
| 순서 | 흐리게 업샘플 → 강조 | **질문 → 창에서 선택** (흐리기 전) |
| detail | 이미 손실 | **자리별 선택으로 복원** |
| 맥락 | 단순 채널/공간 gate | **q·k local attention** (질문↔후보 비교) |
| 겨냥 갭 | 의미(이미 있음→무효) | **흐림/정렬**(미검증 레버) |

---

## 6. 검증 (스모크 테스트)

- **수식==코드** allclose=True (guided local attention 독립 재계산) ✅
- attention 합=1(softmax) ✅ / **γ=0 → out==원본 x0**(안전초기화) ✅
- gradient 흐름·유한성 ✅ / YOLO26s 빌드+E2E(one2many/one2one) 정상, strides [4,8,16,32] ✅
- 비용: params 9.84M, **GFLOPs 28.61** (naive-P2 27.79, **+3%**). ⚠️ local attention unfold로 메모리↑ → **batch 32 OOM 시 16**.

---

## 7. 실험 계획

`yolo26s-p2-gsa.yaml` (SGI-V3와 **동일 입력** = 메커니즘만 다름 → 깨끗한 비교).
```bash
yolo detect train model=".../cfg/models/26/yolo26s-p2-gsa.yaml" \
  data=".../dataset/VisDrone/cleaning/data.yaml" \
  epochs=300 patience=50 batch=16 imgsz=640 device=<2개> workers=4 seed=0 \
  project=".../runs/FPN" name="260701_Y26S_GSA_VISDRONE"
```
**성공 판정 (naive-P2·SGI-V3 대비):** tiny AP50/recall 상승 = "흐림이 병목이었다" 입증.
γ·src_w 점검: `python eval/inspect_sgi.py .../best.pt` (GSA도 gamma/src_w 있음).

> **정직:** 의미는 top-down이 이미 옮겼으니, GSA가 이기려면 **"흐림 제거가 실제 도움"**이어야 함(미검증=해볼 가치). 안 되면 "선명함은 병목이 아니었다"는 깨끗한 음성 결론.

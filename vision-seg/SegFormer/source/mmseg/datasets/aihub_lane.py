from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class AIHubLaneDataset(CustomDataset):
    """AI Hub 차량 전방 차선 데이터셋 (실선/점선 3클래스).

    클래스는 SkyScapesLaneDataset 과 같지만 **이미지 확장자가 .jpg** 라 별도 클래스가
    필요하다. mmseg CustomDataset 은 ``img.replace(img_suffix, seg_map_suffix)`` 로
    라벨 경로를 만들기 때문에 확장자가 정확해야 한다.

      0 background : 그 외 전부 (crosswalk, stop_line 포함)
      1 solid      : traffic_lane 중 lane_type=solid
      2 dashed     : traffic_lane 중 lane_type=dotted

    원본 라벨은 차선 **중심선 폴리라인**이라 두께가 없다. ``UTIL/aihub_make_masks.py`` 가
    y 좌표별 실측 도색 폭(원근 보정)으로 칠해 마스크를 만든다.

    **주의 — SkyScapes 와 라벨의 의미가 다르다.** 폴리라인이 점선의 빈칸을 관통하므로
    dashed 라벨은 도색 픽셀이 아니라 "차선이 지나가는 띠"를 뜻한다. 두 데이터셋의 IoU 를
    직접 비교하면 안 된다.

    ``reduce_zero_label`` 은 반드시 False 다. 0 이 실제 background 클래스다.
    """

    CLASSES = ('background', 'solid', 'dashed')

    PALETTE = [[0, 0, 0], [0, 0, 255], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(AIHubLaneDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

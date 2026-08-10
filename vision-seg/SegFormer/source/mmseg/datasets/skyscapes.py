from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SkyScapesLaneDataset(CustomDataset):
    """DLR-SkyScapes 항공영상 차선 데이터셋 (3클래스로 병합).

    원본 SkyScapes-Lane 13클래스를 실선/점선 판정용으로 병합한 것이다.
      0 background : 원본 0, 4~12 (차선이 아닌 노면표시 전부 포함)
      1 solid      : 원본 2  (Long Line)
      2 dashed     : 원본 1, 3 (Dash Line, Small dash line)

    타일 생성은 ``UTIL/skyscapes_make_tiles.py`` 가 담당하며 결과는 다음 구조다.
      {data_root}/{split}/images/*.png   1024x1024 RGB
      {data_root}/{split}/labels/*.png   1024x1024 uint8, 값 0/1/2 (+ val/test 는 패딩 255)

    val/test 타일의 오른쪽·아래 패딩 픽셀은 255(=ignore_index)라 손실과 mIoU 집계에서
    자동으로 빠진다. 그래서 타일 합계 전경 픽셀 수가 원본과 정확히 일치한다.

    ``reduce_zero_label`` 은 반드시 False 다. True 면 모든 라벨값을 1씩 빼서 0을 ignore 로
    만드는데, 여기서는 0이 실제 background 클래스이기 때문이다.
    """

    CLASSES = ('background', 'solid', 'dashed')

    PALETTE = [[0, 0, 0], [0, 0, 255], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(SkyScapesLaneDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

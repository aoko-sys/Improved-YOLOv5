import utils.autoanchor as autoAC


# 对数据集重新计算anchor
new_anchors = autoAC.kmean_anchors('./data/coco128.yaml',9,640,5.0,2000,True)
print(new_anchors)
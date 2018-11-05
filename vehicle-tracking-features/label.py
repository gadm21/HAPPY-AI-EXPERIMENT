from utils import label_map_util

class Label(object):
    def __init__(self,PATH_TO_LABELS):
        self.PATH_TO_LABELS = PATH_TO_LABELS
        self.category_index = None

    def setup(self):
        NUM_CLASSES = 90

        # Loading label map
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True
        )
        self.category_index = label_map_util.create_category_index(categories)

    def getLabel(self,objclass):
        return self.category_index[objclass]["name"]
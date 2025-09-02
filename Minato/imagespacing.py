import os
import nibabel

class image_spacing_ext:
    def __init__(self,path):

        self.path= path
        self.image_spacing = {}

    def __call__(self):
       
        print(f"Format conversion {self.path}")
        for case in sorted([f for f in os.listdir(self.path) if "case" in f]):
            case_id = int(case.split("_")[1])
            image = nibabel.load(os.path.join(self.path, case, "imaging.nii.gz"))
            image_spacings = image.header["pixdim"][1:4].tolist()
            self.image_spacing[case]= image_spacings
        return self.image_spacing

         
            
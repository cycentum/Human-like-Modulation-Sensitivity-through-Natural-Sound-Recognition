from pathlib import Path

DIR_PROJECT=Path(r"../results") #You can download its contents from the release
DIR_TRAINING=DIR_PROJECT/"Training"
DIR_NET=DIR_TRAINING/"Net"
DIR_TIME_AVE=DIR_PROJECT/"TimeAve"
DIR_TEMPLATE_CORREL=DIR_PROJECT/"TemplateCorrel"
DIR_HUMAN=DIR_PROJECT/"Human"
DIR_PHYSIOLOGY=DIR_PROJECT/"Neurophysiology"

DIR_DATASET_ESC=Path(r"Path to ESC-50") #Set this depending on your environment. It should contain directories named "audio" and "meta"
DIR_DATASET_TIMIT=Path(r"Path to LDC93S1") #Set this depending on your environment. It should contain a directory named "timit" which in turn contains "TIMIT"

DIR_REPO_NEUROPHYSIOLOGY=Path(r"Path to git\cascaded-am-tuning-for-sound-recognition") #Set this depending on your environment. It should contain a directory named "cascaded-am-tuning-for-sound-recognition" which in turn contains python files and other directories.

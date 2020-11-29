from PIL import Image  # Python Image Library - Image Processing
import glob
import os
import shutil

annorealdir_src = "D:/KULIAH/Dataset Sample/Gambar/VOC/VOCdevkit/Training/imagesfixx/annotationsfix/Train"
annorealdir_dst = 'D:/KULIAH/Dataset Sample/Gambar/VOC/VOCdevkit/Training/imagesfixx/annotationsfix/Anno 50-50'
imagerealdir_src = "D:/KULIAH/Dataset Sample/Gambar/VOC/VOCdevkit/Training/imagesfixx/imagesfix_klasifikasi/Train 50-50/Train"
#imagerealdir_dst = 'D:/KULIAH/Dataset Sample/Gambar/VOC/VOCdevkit/VOC2012/imagescar'

# based on SO Answer: https://stackoverflow.com/a/43258974/5086335
'''
for file in glob.glob("*.JPEG"):
 im = Image.open(file)
 rgb_im = im.convert('RGB')
 rgb_im.save(file.replace("JPEG", "png"), quality=95)
'''

for filename in os.listdir(imagerealdir_src):
	#print(filename)
	if filename.endswith('png'):
		try:
			PathFileName = "{}/{}".format(imagerealdir_src, filename)
			#print(PathFileName)
			if os.stat(PathFileName).st_size > 0:
				file_prefix = filename.split(".png")[0]
				#print(file_prefix)
				anno_file_name = "{}.{}".format(file_prefix,"xml")
				print(anno_file_name)
				annopath = '{}/{}'.format(annorealdir_src, anno_file_name)
				#print(annopath)
				shutil.copy(annopath, annorealdir_dst)
				print("Copying success!")
		except:
			print("No")
	else:
		print("Skipping file: {}".format(filename))
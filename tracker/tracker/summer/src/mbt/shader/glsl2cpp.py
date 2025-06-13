import sys
import os

def find_glsl(path):
	dirs = os.listdir(path)
	glsls = []
	for i in dirs:
		if os.path.splitext(i)[1] == ".glsl":
			glsls.append(i)
	return glsls

def strfy(str):
	return '\"' + str + '\\n\"'

def insert(original, new, pos):
	return original[:pos] + new + original[pos:]

def glsl2cppF(glsl_file):
	with open(glsl_file, 'r') as f:
		lines = f.readlines()

	cpp_file = glsl_file[0:-4]+"hh"
	with open("shaders.hh",'w') as f:
		f. write(glsl_file[0:-5] + " = \n")
		for line in lines:
			line = line.strip('\n')
			str = strfy(line)
			print str
			f.write(str+'\n')

def glsl2cpp(glsl_file, content):
	with open(glsl_file, 'r') as f:
		lines = f.readlines()

		content.append('static char ' + glsl_file[0:-5] + "[] = \n")
		for line in lines:
			line = line.strip('\n')
			str = strfy(line)
			content.append(str+'\n')
		content.append(';\n\n')

""" main here """
glsls = find_glsl('./')

content = []
for glsl in glsls:
		glsl2cpp(glsl, content)

with open("shaders.hh",'w') as f:
	f.writelines(content)
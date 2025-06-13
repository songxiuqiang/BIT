#version 330

uniform mat4 uMVMatrix;
uniform mat4 uMVPMatrix;
uniform mat3 uNormalMatrix;

in vec3 aPosition;
in vec3 aNormal;

out vec3 vNormal;

void main() {
	vNormal = normalize(uNormalMatrix * aNormal);
	
	// vertex position
	gl_Position = uMVPMatrix * vec4(aPosition, 1.0);
}
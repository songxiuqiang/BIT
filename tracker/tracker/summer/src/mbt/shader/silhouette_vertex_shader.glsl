#version 330

uniform mat4 uMVPMatrix;
in vec3 aPosition;


void main()
{
	// vertex position
	gl_Position = uMVPMatrix * vec4(aPosition, 1.0);
}
#version 330

in vec3 aPosition;
in vec3 aNormal;
in vec2 aTexCoords;

out vec2 TexCoords;

uniform mat4 uMVPMatrix;

void main()
{
    TexCoords = aTexCoords;
	gl_Position = uMVPMatrix * vec4(aPosition, 1.0);
}
#version 330

uniform vec3 uColor;
uniform float uAlpha;

layout(location = 0) out vec4 fragColor;

void main()
{
	fragColor = vec4(uColor, uAlpha);
}
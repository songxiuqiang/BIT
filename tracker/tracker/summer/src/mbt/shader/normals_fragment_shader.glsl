#version 330

uniform float uAlpha;

in vec3 vNormal;

layout(location = 0) out vec4 fragColor;

void main() {
	vec3 normal = normalize(vNormal);
	fragColor = vec4((vNormal+1.0)/2.0, uAlpha);
}
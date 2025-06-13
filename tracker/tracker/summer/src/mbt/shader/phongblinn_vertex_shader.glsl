#version 330

uniform mat4 uMVMatrix;
uniform mat4 uMVPMatrix;
uniform mat3 uNormalMatrix;
uniform vec3 uLightPosition1;
uniform vec3 uLightPosition2;
uniform vec3 uLightPosition3;

in vec3 aPosition;
in vec3 aNormal;

out vec3 vPosition;
out vec3 vLightPosition1;
out vec3 vLightPosition2;
out vec3 vLightPosition3;
out vec3 vNormal;

void main() {

	vec4 position = uMVMatrix * vec4(aPosition, 1.0);
	
    vec3 normal = normalize(uNormalMatrix * aNormal);

	vPosition = position.xyz;
    vLightPosition1 = uLightPosition1.xyz;
    vLightPosition2 = uLightPosition2.xyz;
    vLightPosition3 = uLightPosition3.xyz;
    
	vNormal = normal;
	
	// vertex position
	gl_Position = uMVPMatrix * vec4(aPosition, 1.0);
}
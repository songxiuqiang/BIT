#version 330 core

in vec2 TexCoords;

layout(location = 0) out vec4 fragColor;

uniform sampler2D texture_diffuse;

void main()
{
    fragColor = texture(texture_diffuse, TexCoords);
}
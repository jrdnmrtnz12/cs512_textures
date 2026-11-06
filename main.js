const VERTEX_SHADER = `#version 300 es
in vec3 aPosition;
in vec3 aNormal;
in vec3 aTangent;
in vec2 aTexCoord;

uniform mat4 uModelView;
uniform mat4 uProjection;

out vec3 vPosition;
out vec3 vNormal;
out vec3 vTangent;
out vec2 vTexCoord;

void main() {
  vec4 p = uModelView * vec4(aPosition, 1.0);
  vPosition = p.xyz;

  // Inverse transpose for potential non-uniform scalings
  mat3 mv = mat3(uModelView);
  vNormal = transpose(inverse(mv)) * aNormal;


  vTangent = mv * aTangent;
  vTexCoord = aTexCoord;
  gl_Position = uProjection * p;
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision mediump float;

in vec3 vPosition;
in vec3 vNormal;
in vec3 vTangent;
in vec2 vTexCoord;

uniform vec3 uLightPos;
uniform sampler2D uTexture;
uniform int uMapping;  // 0 = bump mapping, 1 = normal mapping
// #2: 2 = plain texture with no bumps, used for the video
// #1: 3 = just the image with no lighting, used for the night sky
uniform vec3 uNormalColor;  // #3: the color of a normal mapped part, since its texture only stores normals

out vec4 fragColor;

const float bumpScale = 3.0;

// convert RGB to luminance (Rec._601 standard, https://en.wikipedia.org/wiki/Rec._601)
// or you can use a height map texture
float height(vec2 uv) {
  vec3 c = texture(uTexture, uv).rgb;
  return 1.0 - dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
  // #1: the sky is seen from inside its ball, so it skips the lighting and just shows the image
  if (uMapping == 3) {
    fragColor = vec4(texture(uTexture, vTexCoord).rgb, 1.0);
    return;
  }

  // Tangent frame (T, B, N)
  vec3 N = normalize(vNormal);
  vec3 T = normalize(vTangent - dot(vTangent, N) * N);
  vec3 B = cross(N, T);
  mat3 TBN = mat3(T, B, N);

  vec3 baseColor;
  vec3 n;

  if (uMapping == 0) {
    // Bump mapping: tilt the normal against the gradient of the height field
    vec2 du = vec2(1.0 / float(textureSize(uTexture, 0).x), 0.0);
    vec2 dv = vec2(0.0, 1.0 / float(textureSize(uTexture, 0).y));
    float dhdu = (height(vTexCoord + du) - height(vTexCoord - du)) * 0.5;
    float dhdv = (height(vTexCoord + dv) - height(vTexCoord - dv)) * 0.5;
    n = normalize(vec3(-bumpScale * dhdu, -bumpScale * dhdv, 1.0));
    baseColor = texture(uTexture, vTexCoord).rgb;
  } else if (uMapping == 2) {
    // #2: plain texture. (0, 0, 1) leaves the normal as it is, and the color comes from the texture
    n = vec3(0.0, 0.0, 1.0);
    baseColor = texture(uTexture, vTexCoord).rgb;
  } else {
    // Normal mapping: the texture stores the tangent-space normal in [0, 1]
    n = normalize(texture(uTexture, vTexCoord).rgb * 2.0 - 1.0);
    baseColor = uNormalColor;
  }

  vec3 Ka = 0.2 * baseColor;
  vec3 Kd = 0.8 * baseColor;
  vec3 Ks = vec3(0.5);
  float shininess = 32.0;

  N = normalize(TBN * n);
  vec3 L = normalize(uLightPos - vPosition);
  vec3 V = normalize(-vPosition);
  vec3 R = reflect(-L, N);

  float diff = max(dot(N, L), 0.0);
  float spec = pow(max(dot(V, R), 0.0), shininess);

  fragColor = vec4(Ka + Kd * diff + Ks * spec, 1.0);
}
`;

// WebGL setup
const canvas = document.getElementById("glcanvas");
const gl = canvas.getContext("webgl2");

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

const ATTRIBUTES = { aPosition: 0, aNormal: 1, aTexCoord: 2, aTangent: 3 };

function createProgram(vsSource, fsSource) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSource));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSource));
  for (const name in ATTRIBUTES) gl.bindAttribLocation(prog, ATTRIBUTES[name], name);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  return prog;
}

// Shader editors
const vertEditor = document.getElementById("vertEditor");
const fragEditor = document.getElementById("fragEditor");
const shaderError = document.getElementById("shaderError");
vertEditor.value = VERTEX_SHADER;
fragEditor.value = FRAGMENT_SHADER;

let program, uModelView, uProjection, uLightPos, uMapping;
let uNormalColor;

function updateProgram() {
  try {
    const prog = createProgram(vertEditor.value, fragEditor.value);
    gl.deleteProgram(program);
    program = prog;
    gl.useProgram(program);
    uModelView = gl.getUniformLocation(program, "uModelView");
    uProjection = gl.getUniformLocation(program, "uProjection");
    uLightPos = gl.getUniformLocation(program, "uLightPos");
    uMapping = gl.getUniformLocation(program, "uMapping");
    uNormalColor = gl.getUniformLocation(program, "uNormalColor");
    gl.uniform1i(gl.getUniformLocation(program, "uTexture"), 0);
    shaderError.textContent = "";
  } catch (e) {
    shaderError.textContent = e.message.replace(/\0/g, "");
  }
}
updateProgram();
vertEditor.addEventListener("input", updateProgram);
fragEditor.addEventListener("input", updateProgram);

function bindAttribute(name, data, size) {
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.vertexAttribPointer(ATTRIBUTES[name], size, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(ATTRIBUTES[name]);
}

function createMesh({ positions, normals, tangents, texCoords, indices }) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  bindAttribute("aPosition", positions, 3);
  bindAttribute("aNormal", normals, 3);
  bindAttribute("aTangent", tangents, 3);
  bindAttribute("aTexCoord", texCoords, 2);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  return { vao, count: indices.length };
}

const cubeMesh = createMesh(createCube());
const cylinderMesh = createMesh(createCylinder());

// #1: the three new primitives: cone, sphere and torus. the image wraps around the sphere twice,
// so the whole picture fits on the half that faces the camera
const coneMesh = createMesh(createCone());
const sphereMesh = createMesh(createSphere(64, 32, 2));
const torusMesh = createMesh(createTorus());

function loadTexture(url) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 255, 255]));

  const image = new Image();
  image.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);  // images start at the top row, texture coordinates at the bottom
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  };
  image.src = url;
  return texture;
}

const brickTexture = loadTexture("brick.png");
const normalTexture = loadTexture("texture2.jpg");
const snowTexture = loadTexture("snow.png");
const carrotTexture = loadTexture("carrot.png");
const buttonTexture = loadTexture("button.png");
const nightTexture = loadTexture("night.png");

// #2: starts the video playing. it has to be muted or the browser won't start it on its own.
// this texture has no mipmaps since they would have to be rebuilt every frame
function loadVideo(url) {
  const video = document.createElement("video");
  video.src = url;
  video.loop = true;
  video.muted = true;
  video.playsInline = true;
  video.play();
  return video;
}
const snowVideo = loadVideo("snowVideo.mp4");
const videoTexture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, videoTexture);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([200, 220, 255, 255]));
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

// #2: copies the video's current frame into its texture. this runs every frame, which is what makes
// the texture move
function updateVideoTexture() {
  if (snowVideo.readyState >= snowVideo.HAVE_CURRENT_DATA) {
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, snowVideo);
  }
}

const BUMP_MAPPING = 0, NORMAL_MAPPING = 1;
const PLAIN_TEXTURE = 2, NO_LIGHTING = 3;

// #1: every part of the snowman. offset is where it sits, size is how much it's stretched on each axis,
// and turn is how it's rotated (only some parts need one). color only matters for normal mapped parts
const NO_TURN = rotateZ(0);
const ARM_LENGTH = 0.78;
const GOLD = [0.85, 0.65, 0.3], HAT_BLACK = [0.06, 0.06, 0.07];
const objects = [
  // #1: the night sky, a huge ball around the whole scene with the star image on it
  { mesh: sphereMesh, texture: nightTexture, mapping: NO_LIGHTING, offset: [0, 0, 0], size: [40, 40, 40] },

  // #1: the skeleton's cube, flattened into the brick platform the snowman stands on
  { mesh: cubeMesh, texture: brickTexture, mapping: BUMP_MAPPING, offset: [0, -2.3, 0], size: [2.2, 0.2, 1.6] },

  // #3: bump mapping. the body and head are both balls with the snow image
  { mesh: sphereMesh, texture: snowTexture, mapping: BUMP_MAPPING, offset: [0, -1, 0], size: [1.1, 1.1, 1.1] },
  { mesh: sphereMesh, texture: snowTexture, mapping: BUMP_MAPPING, offset: [0, 0.7, 0], size: [0.7, 0.7, 0.7] },

  // #1: the skeleton's cylinder as the top hat, plus a wide flat one for the brim
  { mesh: cylinderMesh, texture: normalTexture, mapping: NORMAL_MAPPING, offset: [0, 1.85, 0], size: [0.45, 0.5, 0.45], color: HAT_BLACK },
  { mesh: cylinderMesh, texture: normalTexture, mapping: NORMAL_MAPPING, offset: [0, 1.33, 0], size: [0.72, 0.05, 0.72], color: HAT_BLACK },

  // #1: the button eyes, flat cylinders with the button image on the cap that faces out.
  // each turn tips the cap so it lies flat against the head at that spot
  { mesh: cylinderMesh, texture: buttonTexture, mapping: BUMP_MAPPING, offset: [-0.255, 0.853, 0.634], size: [0.12, 0.015, 0.12], turn: multiply(rotateZ(1.03), rotateX(1.13)) },
  { mesh: cylinderMesh, texture: buttonTexture, mapping: BUMP_MAPPING, offset: [0.255, 0.853, 0.634], size: [0.12, 0.015, 0.12], turn: multiply(rotateZ(-1.03), rotateX(1.13)) },

  // #1: the carrot nose. the cone's tip points up, so a 90 degree turn about x points it forward
  { mesh: coneMesh, texture: carrotTexture, mapping: BUMP_MAPPING, offset: [0, 0.7, 0.85], size: [0.12, 0.25, 0.12], turn: rotateX(Math.PI / 2) },

  // #1: the stick arms, thin cylinders angled from the shoulders down to the ring
  { mesh: cylinderMesh, texture: brickTexture, mapping: BUMP_MAPPING, offset: [0.81, -0.82, 1], size: [0.05, ARM_LENGTH, 0.05], turn: multiply(rotateZ(-0.47), rotateX(-1)) },
  { mesh: cylinderMesh, texture: brickTexture, mapping: BUMP_MAPPING, offset: [-0.81, -0.82, 1], size: [0.05, ARM_LENGTH, 0.05], turn: multiply(rotateZ(0.47), rotateX(-1)) },

  // #2: the snow globe, a ball with the video on it. the 90 degree turn about y moves the edge of the
  // video to the sides, and the extra 0.5 undoes dragRotY's start so the middle of it faces the camera
  { mesh: sphereMesh, texture: videoTexture, mapping: PLAIN_TEXTURE, offset: [0, -0.85, 1.75], size: [0.6, 0.6, 0.6], turn: rotateY(Math.PI / 2 + 0.5) },

  // #3: normal mapping. the donut under the globe that holds it up
  { mesh: torusMesh, texture: normalTexture, mapping: NORMAL_MAPPING, offset: [0, -1.3, 1.75], size: [0.55, 0.55, 0.55], color: GOLD }
];

gl.enable(gl.DEPTH_TEST);
gl.clearColor(0, 0, 0, 1);
const projection = perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100);
const lightWorld = [3, 4, 5];

// Interactive controls
let dragRotX = 0, dragRotY = -0.5;      // mouse drag
let camX = 0, camY = 0.2, camZ = -10;   // arrow keys, w, s
// #1: camY and camZ moved so the whole snowman and platform fit in the canvas with room around them
let sliderAngle = 0;                    // degrees
let spinning = false, spinAngle = 0;    // Start/Stop button

let mouseDown = false, lastX, lastY;
canvas.addEventListener("mousedown", e => { mouseDown = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", () => mouseDown = false);
window.addEventListener("mousemove", e => {
  if (!mouseDown) return;
  dragRotY += (e.clientX - lastX) * 0.01;
  dragRotX += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
});

document.addEventListener("keydown", e => {
  if (e.target === vertEditor || e.target === fragEditor) return;
  const step = 0.2;
  switch (e.key) {
    case "ArrowUp": camY -= step; break;
    case "ArrowDown": camY += step; break;
    case "ArrowLeft": camX += step; break;
    case "ArrowRight": camX -= step; break;
    case "w": camZ += step; break;
    case "s": camZ -= step; break;
    default: return;
  }
  e.preventDefault();
});

const axisSelect = document.getElementById("axisSelect");
document.getElementById("slider").addEventListener("input", e => {
  sliderAngle = Number(e.target.value);
  document.getElementById("sliderValue").textContent = sliderAngle;
});
document.getElementById("spinButton").addEventListener("click", () => spinning = !spinning);

// Render loop
let lastTime = 0;
function render(time) {
  const t = time / 1000;
  if (spinning) spinAngle += t - lastTime;
  lastTime = t;
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const view = translate(camX, camY, camZ);
  gl.uniformMatrix4fv(uProjection, false, projection);
  gl.uniform3fv(uLightPos, transformPoint(view, lightWorld));

  const axisRotation = ROTATE[axisSelect.value](sliderAngle * Math.PI / 180 + spinAngle);
  const rotation = multiply(multiply(rotateY(dragRotY), rotateX(dragRotX)), axisRotation);

  gl.activeTexture(gl.TEXTURE0);
  updateVideoTexture();
  for (const { mesh, texture, mapping, offset, size, turn = NO_TURN, color = GOLD } of objects) {
    // #1: M_part = T * R * S puts the part in the scene, and the rotation goes on the left of that
    // so the whole snowman turns as one piece instead of each part spinning on its own
    const part = multiply(translate(...offset), multiply(turn, scale(...size)));
    gl.uniformMatrix4fv(uModelView, false, multiply(view, multiply(rotation, part)));
    gl.uniform1i(uMapping, mapping);
    gl.uniform3fv(uNormalColor, color);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.bindVertexArray(mesh.vao);
    gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
  }
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

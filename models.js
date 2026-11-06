function createCube() {
  const positions = new Float32Array([
    -1, -1,  1,   1, -1,  1,   1,  1,  1,  -1,  1,  1,   // front
     1, -1, -1,  -1, -1, -1,  -1,  1, -1,   1,  1, -1,   // back
    -1,  1,  1,   1,  1,  1,   1,  1, -1,  -1,  1, -1,   // top
    -1, -1, -1,   1, -1, -1,   1, -1,  1,  -1, -1,  1,   // bottom
     1, -1,  1,   1, -1, -1,   1,  1, -1,   1,  1,  1,   // right
    -1, -1, -1,  -1, -1,  1,  -1,  1,  1,  -1,  1, -1    // left
  ]);
  const faceNormals = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]];
  const normals = new Float32Array(faceNormals.flatMap(n => [...n, ...n, ...n, ...n]));

  // Tangent; bitangent = cross(N, T)
  const faceTangents = [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]];
  const tangents = new Float32Array(faceTangents.flatMap(t => [...t, ...t, ...t, ...t]));

  const texCoords = new Float32Array(faceNormals.flatMap(() => [0, 0, 1, 0, 1, 1, 0, 1]));

  const indices = new Uint16Array([
     0,  1,  2,   0,  2,  3,   // front
     4,  5,  6,   4,  6,  7,   // back
     8,  9, 10,   8, 10, 11,   // top
    12, 13, 14,  12, 14, 15,   // bottom
    16, 17, 18,  16, 18, 19,   // right
    20, 21, 22,  20, 22, 23    // left
  ]);
  return { positions, normals, tangents, texCoords, indices };
}


function createCylinder(segments = 64, uRepeat = 3) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];

  // Side: theta grows with u, so T points along the circle and B = cross(N, T) points up
  for (let i = 0; i <= segments; i++) {
    const u = i / segments, theta = 2 * Math.PI * u;
    const s = Math.sin(theta), c = Math.cos(theta);
    for (const y of [-1, 1]) {
      positions.push(s, y, c);
      normals.push(s, 0, c);
      tangents.push(c, 0, -s);
      texCoords.push(u * uRepeat, (y + 1) / 2);
    }
  }
  for (let i = 0; i < segments; i++) {
    const k = 2 * i;
    indices.push(k, k + 2, k + 3, k, k + 3, k + 1);
  }

  // Caps: planar mapping of x, z into the texture, T = +x on both
  for (const y of [1, -1]) {
    const center = positions.length / 3;
    positions.push(0, y, 0);
    normals.push(0, y, 0);
    tangents.push(1, 0, 0);
    texCoords.push(0.5, 0.5);
    for (let i = 0; i <= segments; i++) {
      const theta = 2 * Math.PI * i / segments;
      const s = Math.sin(theta), c = Math.cos(theta);
      positions.push(s, y, c);
      normals.push(0, y, 0);
      tangents.push(1, 0, 0);
      texCoords.push(0.5 + 0.5 * s, 0.5 - 0.5 * y * c);  // keeps cross(N, T) along +v
    }
    for (let i = 0; i < segments; i++) {
      const a = center + 1 + i, b = a + 1;
      if (y > 0) indices.push(center, a, b);
      else indices.push(center, b, a);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}

// #1: cone with its tip at y = 1 and a base of radius 1 at y = -1 (so h = 2, r = 1).
// the side normal leans up because the side slants: (h sin, r, h cos) / sqrt(r^2 + h^2)
function createCone(segments = 64) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];
  const h = 2, r = 1, len = Math.sqrt(r * r + h * h);

  // #1: side. u goes around and v goes from the base (0) up to the tip (1).
  // each step adds a base point and a tip point, and the tip uses the middle of the slice
  for (let i = 0; i <= segments; i++) {
    const u = i / segments, theta = 2 * Math.PI * u, mid = theta + Math.PI / segments;
    const s = Math.sin(theta), c = Math.cos(theta);
    const sMid = Math.sin(mid), cMid = Math.cos(mid);
    positions.push(s, -1, c, 0, 1, 0);
    normals.push(h * s / len, r / len, h * c / len, h * sMid / len, r / len, h * cMid / len);
    tangents.push(c, 0, -s, cMid, 0, -sMid);  // #3: the tangent points the way u grows, same as the cylinder
    texCoords.push(u, 0, u + 0.5 / segments, 1);
  }
  for (let i = 0; i < segments; i++) {
    const k = 2 * i;
    indices.push(k, k + 2, k + 1);
  }

  // #1: base cap facing down, with the same flat mapping as the cylinder's bottom cap
  const center = positions.length / 3;
  positions.push(0, -1, 0);
  normals.push(0, -1, 0);
  tangents.push(1, 0, 0);
  texCoords.push(0.5, 0.5);
  for (let i = 0; i <= segments; i++) {
    const theta = 2 * Math.PI * i / segments;
    const s = Math.sin(theta), c = Math.cos(theta);
    positions.push(s, -1, c);
    normals.push(0, -1, 0);
    tangents.push(1, 0, 0);
    texCoords.push(0.5 + 0.5 * s, 0.5 + 0.5 * c);
  }
  for (let i = 0; i < segments; i++) {
    const a = center + 1 + i, b = a + 1;
    indices.push(center, b, a);
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}

// #1: ball of radius 1. phi goes from the top to the bottom and theta goes around, and there's a point
// at each step. on a ball of radius 1 the point is also its normal, since it points out from the center.
// uRepeat is how many times the image wraps around the ball
function createSphere(uSegments = 64, vSegments = 32, uRepeat = 1) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];

  for (let i = 0; i <= vSegments; i++) {
    const phi = i * Math.PI / vSegments;
    for (let j = 0; j <= uSegments; j++) {
      const u = j / uSegments, theta = 2 * Math.PI * u;
      const s = Math.sin(theta), c = Math.cos(theta);
      const x = s * Math.sin(phi), y = Math.cos(phi), z = c * Math.sin(phi);
      positions.push(x, y, z);
      normals.push(x, y, z);
      tangents.push(c, 0, -s);  // #3: the tangent points the way u grows
      texCoords.push(u * uRepeat, 1 - i / vSegments);  // #1: v is 1 at the top and 0 at the bottom
    }
  }

  // #1: connects each point to its neighbors (one step over, one row down) to make the triangles
  for (let i = 0; i < vSegments; i++) {
    for (let j = 0; j < uSegments; j++) {
      const k1 = i * (uSegments + 1) + j;
      const k2 = k1 + uSegments + 1;
      indices.push(k1, k2, k1 + 1);
      indices.push(k2, k2 + 1, k1 + 1);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}

// #1: donut lying flat. R is the big ring's radius and r is the tube's radius. theta goes around the
// big ring and phi goes around the tube. the normal points out from the middle of the tube:
// (cos phi sin theta, sin phi, cos phi cos theta)
function createTorus(R = 1, r = 0.28, uSegments = 64, vSegments = 24, uRepeat = 4) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];

  for (let i = 0; i <= vSegments; i++) {
    const v = i / vSegments, phi = 2 * Math.PI * v;
    const sinp = Math.sin(phi), cosp = Math.cos(phi);
    for (let j = 0; j <= uSegments; j++) {
      const u = j / uSegments, theta = 2 * Math.PI * u;
      const s = Math.sin(theta), c = Math.cos(theta);
      positions.push((R + r * cosp) * s, r * sinp, (R + r * cosp) * c);
      normals.push(cosp * s, sinp, cosp * c);
      tangents.push(c, 0, -s);  // #3: the tangent points the way u grows
      texCoords.push(u * uRepeat, v);  // #1: u goes around the big ring and v goes around the tube
    }
  }

  // #1: same neighbor connecting as the ball
  for (let i = 0; i < vSegments; i++) {
    for (let j = 0; j < uSegments; j++) {
      const k1 = i * (uSegments + 1) + j;
      const k2 = k1 + uSegments + 1;
      indices.push(k1, k2, k1 + 1);
      indices.push(k2, k2 + 1, k1 + 1);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}

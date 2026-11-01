import React, { useEffect, useRef } from "react";
import * as THREE from "three";

// Plain three.js (no react-three-fiber dependency) rendering token
// embeddings as a rotating point cloud with labels via sprites.
export default function EmbeddingSpace({ latest }) {
  const mountRef = useRef(null);
  const stateRef = useRef({});

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(0, 1.5, 6);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    mount.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    // faint grid floor for spatial reference, in the theme's line colors
    // (three.js can't read CSS variables, so resolve them here)
    const css = getComputedStyle(mount);
    const grid = new THREE.GridHelper(8, 16, css.getPropertyValue("--line-strong").trim(), css.getPropertyValue("--line").trim());
    scene.add(grid);

    stateRef.current = { scene, camera, renderer, group, points: null, labels: [] };

    function resize() {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (!w || !h) return;
      // updateStyle=true (default) so the canvas CSS size matches its container;
      // passing false leaves the buffer at w*dpr in CSS px and overflows the panel.
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    let raf;
    function animate() {
      group.rotation.y += 0.0025;
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    }
    animate();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  // update points whenever new embeddings arrive
  useEffect(() => {
    const { group } = stateRef.current;
    if (!group || !latest?.embeddings) return;

    // clear old, freeing their GPU buffers (a new cloud arrives every frame)
    while (group.children.length) {
      const old = group.children[0];
      group.remove(old);
      old.geometry.dispose();
      old.material.dispose();
    }
    const accent = getComputedStyle(mountRef.current).getPropertyValue("--accent").trim();

    const pts = latest.embeddings;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(pts.length * 3);
    pts.forEach((p, i) => {
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
    });
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
      color: accent,
      size: 0.12,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
    });
    group.add(new THREE.Points(geo, mat));

    // connective lines to origin-ish centroid to suggest a manifold, subtle
    const lineGeo = new THREE.BufferGeometry();
    const linePos = [];
    for (let i = 0; i < pts.length - 1; i++) {
      linePos.push(pts[i].x, pts[i].y, pts[i].z, pts[i + 1].x, pts[i + 1].y, pts[i + 1].z);
    }
    lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(linePos, 3));
    const lineMat = new THREE.LineBasicMaterial({ color: accent, transparent: true, opacity: 0.3 });
    group.add(new THREE.LineSegments(lineGeo, lineMat));
  }, [latest]);

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">Signal / 03</div>
          <div className="title">Embedding Space (3D proj.)</div>
        </div>
        <div className="readout">{latest?.embeddings ? `${latest.embeddings.length} tok` : "—"}</div>
      </div>
      <div className="panel-body" ref={mountRef} />
    </div>
  );
}

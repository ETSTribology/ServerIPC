import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { ColorManagement } from 'three';

// Initialize Socket.io
const socket = io();

// Initialize Three.js variables
let mesh;
let scene, camera, renderer, controls;
let ambientLight, directionalLight;
let isPlaying = false;
let stats;
let gridHelper;
let vertexHelper;
let wireframeHelper;
let composer;
let bloomPass;

// Initialize Three.js
function initThreeJS() {
  console.log('Initializing Three.js');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x333333);  // Set a dark gray background

  camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.set(0, 5, 10);  // Move camera back a bit

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1;
  ColorManagement.enabled = true;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.body.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  ambientLight = new THREE.AmbientLight(0x404040, 0.5);
  scene.add(ambientLight);

  directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
  directionalLight.position.set(1, 1, 1);
  directionalLight.castShadow = true;
  directionalLight.shadow.mapSize.width = 1024;
  directionalLight.shadow.mapSize.height = 1024;
  directionalLight.shadow.camera.near = 1;
  directionalLight.shadow.camera.far = 50;
  scene.add(directionalLight);

  // Create better infinite grid
  createInfiniteGrid();

  // Initialize stats
  stats = new Stats();
  document.body.appendChild(stats.dom);
  stats.dom.style.display = 'none';

  // Set up post-processing
  composer = new EffectComposer(renderer);
  const renderPass = new RenderPass(scene, camera);
  composer.addPass(renderPass);

  bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
  bloomPass.threshold = 0;
  bloomPass.strength = 0.5;
  bloomPass.radius = 0;
  // composer.addPass(bloomPass);

  const outputPass = new OutputPass();
  composer.addPass(outputPass);

  window.addEventListener('resize', onWindowResize, false);

  animate();
  setupControls();
}

function createInfiniteGrid() {
  const size = 100;
  const divisions = 100;
  const centerColor = new THREE.Color(0x444444);
  const gridColor = new THREE.Color(0x888888);

  gridHelper = new THREE.Group();

  const xGrid = new THREE.GridHelper(size, divisions, centerColor, gridColor);
  xGrid.rotation.x = Math.PI / 2;
  gridHelper.add(xGrid);

  const yGrid = new THREE.GridHelper(size, divisions, centerColor, gridColor);
  gridHelper.add(yGrid);

  const zGrid = new THREE.GridHelper(size, divisions, centerColor, gridColor);
  zGrid.rotation.z = Math.PI / 2;
  gridHelper.add(zGrid);

  scene.add(gridHelper);
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  if (stats.dom.style.display !== 'none') {
    stats.update();
  }

  // Update infinite grid
  if (gridHelper) {
    gridHelper.position.set(
      Math.round(camera.position.x / 100) * 100,
      Math.round(camera.position.y / 100) * 100,
      Math.round(camera.position.z / 100) * 100
    );
  }

  composer.render();
  console.log('Rendering frame');  // Add this line for debugging
}

// Function to update mesh in Three.js
function updateMesh(meshData) {
  const { BX, BX_shape, faces, faces_shape } = meshData;

  if (!BX || !faces || !BX_shape || !faces_shape) {
    console.error('Invalid mesh data received');
    return;
  }

  // Convert base64 strings back to ArrayBuffers
  const positionsBuffer = Uint8Array.from(atob(BX), c => c.charCodeAt(0)).buffer;
  const indicesBuffer = Uint8Array.from(atob(faces), c => c.charCodeAt(0)).buffer;

  // Create typed arrays from the buffers
  const positionsArray = new Float64Array(positionsBuffer);
  const indices = new Uint32Array(indicesBuffer);

  // Swap Y and Z coordinates
  const positions = new Float32Array(positionsArray.length);
  for (let i = 0; i < positionsArray.length; i += 3) {
    positions[i] = positionsArray[i];       // X remains the same
    positions[i + 1] = positionsArray[i + 2]; // Y becomes Z
    positions[i + 2] = -positionsArray[i + 1]; // Z becomes -Y (to maintain right-hand coordinate system)
  }

  if (!mesh) {
    const geometry = new THREE.BufferGeometry();
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      color: 0x0077ff,
      shininess: 100,
      side: THREE.DoubleSide,
    });
    mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);

    // Create vertex helper
    createVertexHelper();

    // Create wireframe helper
    createWireframeHelper();

    // Center the mesh only once when it's first created
    centerMesh();
  } else {
    mesh.geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    mesh.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    mesh.geometry.attributes.position.needsUpdate = true;
    mesh.geometry.index.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
    mesh.geometry.computeBoundingSphere();
  }

  // Update vertex helper
  updateVertexHelper();

  // Update wireframe helper
  updateWireframeHelper();
}

function createVertexHelper() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', mesh.geometry.getAttribute('position'));
  const material = new THREE.PointsMaterial({
    color: 0xffff00,
    size: 0.05,
    sizeAttenuation: true
  });
  vertexHelper = new THREE.Points(geometry, material);
  vertexHelper.visible = false;  // Initially invisible, controlled via UI
  mesh.add(vertexHelper); // **Parent to mesh**
}

function updateVertexHelper() {
  if (vertexHelper) {
    vertexHelper.geometry.setAttribute('position', mesh.geometry.getAttribute('position'));
    vertexHelper.geometry.attributes.position.needsUpdate = true;
  }
}

function createWireframeHelper() {
  const material = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 1 });
  wireframeHelper = new THREE.LineSegments(new THREE.WireframeGeometry(mesh.geometry), material);
  wireframeHelper.visible = false;  // Initially invisible, controlled via UI
  mesh.add(wireframeHelper); // **Parent to mesh**
}

function updateWireframeHelper() {
  if (wireframeHelper) {
    wireframeHelper.geometry.dispose();
    wireframeHelper.geometry = new THREE.WireframeGeometry(mesh.geometry);
  }
}

function centerMesh() {
  if (!mesh) return;

  mesh.geometry.computeBoundingBox();
  const boundingBox = mesh.geometry.boundingBox;
  const center = new THREE.Vector3();
  boundingBox.getCenter(center);
  mesh.position.sub(center);

  // Adjust the camera to fit the mesh
  const boundingSphere = mesh.geometry.boundingSphere;
  const radius = boundingSphere.radius;
  const fov = camera.fov * (Math.PI / 180);
  const distance = Math.abs(radius / Math.sin(fov / 2));

  camera.position.set(distance, distance, distance);
  camera.lookAt(scene.position);
  controls.target.set(0, 0, 0);
  controls.update();
}

// Add Socket.IO listener for mesh data
socket.on('meshData', (data) => {
  console.log('Received mesh data:', data);
  try {
    const meshData = JSON.parse(data);
    console.log('Parsed mesh data:', meshData);
    if (isPlaying) {
      updateMesh(meshData);
    }
  } catch (error) {
    console.error('Error parsing mesh data:', error);
  }
});

function setupControls() {
  // Simulation controls
  const startButton = document.getElementById('startSimulation');
  const stopButton = document.getElementById('stopSimulation');
  const pauseButton = document.getElementById('pauseSimulation');
  const playButton = document.getElementById('playSimulation');

  startButton.addEventListener('click', () => {
    fetch('/start-simulation', { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        console.log(data);
        isPlaying = true;
      })
      .catch(error => console.error('Error:', error));
  });

  stopButton.addEventListener('click', () => {
    fetch('/stop-simulation', { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        console.log(data);
        isPlaying = false;
      })
      .catch(error => console.error('Error:', error));
  });

  pauseButton.addEventListener('click', () => {
    isPlaying = false;
  });

  playButton.addEventListener('click', () => {
    isPlaying = true;
  });

  // Light controls
  const ambientIntensitySlider = document.getElementById('ambientLight');
  const directionalIntensitySlider = document.getElementById('directionalLight');

  ambientIntensitySlider.value = ambientLight.intensity;
  directionalIntensitySlider.value = directionalLight.intensity;

  ambientIntensitySlider.addEventListener('input', (event) => {
    ambientLight.intensity = parseFloat(event.target.value);
  });

  directionalIntensitySlider.addEventListener('input', (event) => {
    directionalLight.intensity = parseFloat(event.target.value);
  });

  // Renderer controls
  const showVerticesCheckbox = document.getElementById('showVertices');
  const showWireframeCheckbox = document.getElementById('showWireframe');
  const showFPSCheckbox = document.getElementById('showFPS');
  const infiniteGridCheckbox = document.getElementById('infiniteGrid');

  showVerticesCheckbox.addEventListener('change', (event) => {
    if (vertexHelper) vertexHelper.visible = event.target.checked;
  });

  showWireframeCheckbox.addEventListener('change', (event) => {
    if (wireframeHelper) wireframeHelper.visible = event.target.checked;
  });

  showFPSCheckbox.addEventListener('change', (event) => {
    stats.dom.style.display = event.target.checked ? 'block' : 'none';
  });

  infiniteGridCheckbox.addEventListener('change', (event) => {
    gridHelper.visible = event.target.checked;
  });

  // Scene background color control
  const backgroundColorPicker = document.getElementById('backgroundColor');
  backgroundColorPicker.value = '#000000';

  backgroundColorPicker.addEventListener('input', (event) => {
    scene.background = new THREE.Color(event.target.value);
  });

  // Bloom effect control
  const bloomStrengthSlider = document.getElementById('bloomStrength');
  bloomStrengthSlider.addEventListener('input', (event) => {
    bloomPass.strength = parseFloat(event.target.value);
  });

  // Shadow toggle
  const shadowToggle = document.getElementById('shadowToggle');
  shadowToggle.addEventListener('change', (event) => {
    renderer.shadowMap.enabled = event.target.checked;
    directionalLight.castShadow = event.target.checked;
    if (mesh) {
      mesh.castShadow = event.target.checked;
      mesh.receiveShadow = event.target.checked;
    }
  });

  // SSAO toggle (placeholder)
  const ssaoToggle = document.getElementById('ssaoToggle');
  ssaoToggle.addEventListener('change', (event) => {
  });
}

// Initialize
initThreeJS();

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb); // Sky blue background

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Resize support
window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
});

// Lighting
const ambientLight = new THREE.AmbientLight(0x404040, 1); // Ambient light for soft global illumination
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0xffffff, 1, 100);
pointLight.position.set(10, 15, 10);
scene.add(pointLight);

// Directional light for shadows and sun-like illumination
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(5, 10, 5);
directionalLight.castShadow = true;
scene.add(directionalLight);

// Spot light to simulate a "spotlight" effect on certain areas
const spotLight = new THREE.SpotLight(0xffa500, 1, 100, Math.PI / 4, 1, 2);
spotLight.position.set(-50, 50, -50);
spotLight.target.position.set(0, 0, 0);
scene.add(spotLight);
scene.add(spotLight.target);

// ===== Environment =====
// Ground
const groundGeo = new THREE.PlaneGeometry(500, 500);  // Increased size of ground
const groundMat = new THREE.MeshStandardMaterial({ color: 0x228B22 }); // Forest green
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true; // Make the ground receive shadows
scene.add(ground);

// Box buildings or trees (Increased size of trees)
for (let i = 0; i < 100; i++) {  // Increased number of objects
    const boxGeo = new THREE.BoxGeometry(5, Math.random() * 8 + 5, 5);  // Increased tree size
    const boxMat = new THREE.MeshStandardMaterial({ color: 0x8B4513 }); // Brown
    const box = new THREE.Mesh(boxGeo, boxMat);
    box.position.set(
        Math.random() * 500 - 250,  // Larger spread for objects
        box.geometry.parameters.height / 2,
        Math.random() * 500 - 250  // Larger spread for objects
    );
    box.castShadow = true;  // Objects can cast shadows
    box.receiveShadow = true; // Objects can receive shadows
    scene.add(box);
}

// ===== Drone Load =====
const mtlLoader = new THREE.MTLLoader();
const objLoader = new THREE.OBJLoader();
let drone = null;

mtlLoader.load('drone-blu.mtl', (materials) => {
    materials.preload();
    objLoader.setMaterials(materials);
    objLoader.load('drone.obj', (object) => {
        drone = object;
        drone.scale.set(0.5, 0.5, 0.5);
        drone.position.set(0, 1, 0);
        drone.castShadow = true;
        scene.add(drone);
    });
});

// ===== Controls =====
let moveForward = false, moveBackward = false, moveLeft = false, moveRight = false;
let moveUp = false, moveDown = false;
let zoomSpeed = 2; // Zoom speed factor
let minZoom = 5;   // Minimum camera distance
let maxZoom = 50;  // Maximum camera distance
let zoomDistance = 10;  // Default camera distance from the drone

document.addEventListener('keydown', (e) => {
    if (e.key === 'w') moveForward = true;
    if (e.key === 's') moveBackward = true;
    if (e.key === 'a') moveLeft = true;
    if (e.key === 'd') moveRight = true;
    if (e.key === 'q') moveUp = true;
    if (e.key === 'Shift') moveDown = true;
});

document.addEventListener('keyup', (e) => {
    if (e.key === 'w') moveForward = false;
    if (e.key === 's') moveBackward = false;
    if (e.key === 'a') moveLeft = false;
    if (e.key === 'd') moveRight = false;
    if (e.key === 'q') moveUp = false;
    if (e.key === 'Shift') moveDown = false;
});

// Zoom functionality with mouse wheel
window.addEventListener('wheel', (e) => {
    if (e.deltaY > 0) {
        zoomDistance += zoomSpeed;  // Zoom out
    } else {
        zoomDistance -= zoomSpeed;  // Zoom in
    }

    // Clamp zoom distance to be within the zoom limits
    zoomDistance = Math.max(minZoom, Math.min(maxZoom, zoomDistance));
});

// ===== Animation Loop =====
function animate() {
    requestAnimationFrame(animate);

    if (drone) {
        if (moveForward) drone.position.z -= 0.8;
        if (moveBackward) drone.position.z += 0.8;
        if (moveLeft) drone.position.x -= 0.8;
        if (moveRight) drone.position.x += 0.8;
        if (moveUp) drone.position.y += 0.8;
        if (moveDown) drone.position.y -= 0.8;
    }

    // Camera follows drone, but respects zoom distance
    camera.position.x = drone.position.x;
    camera.position.y = drone.position.y + 5; // Camera 5 units above the drone
    camera.position.z = drone.position.z + zoomDistance; // Camera position adjusted by zoomDistance
    camera.lookAt(drone.position);  // Camera looks at the drone's position

    renderer.render(scene, camera);
}
animate();

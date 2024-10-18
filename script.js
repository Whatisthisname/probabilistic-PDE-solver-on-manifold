(function () {
    let scene, camera, renderer, controls;
    let mesh, meshGeometry, meshMaterial;
    let meshData; // To store the loaded mesh data including color profiles
    let colorScale; // To store the current color scale
    let currentMinScalar, currentMaxScalar; // To store current min and max scalar values
    let currentProfileName; // To store the current profile name
    let currentTimeIndex; // For time-dependent profiles

    function init() {
        // Create the scene and set the camera
        scene = new THREE.Scene();
        // load image from url and set to background:
        const loader = new THREE.TextureLoader();

        viewport = document.getElementById('grid-block-mesh-viewport');


        loader.load("https://static.vecteezy.com/system/resources/previews/003/659/551/original/abstract-black-and-white-grid-striped-geometric-seamless-pattern-illustration-free-vector.jpg", function (texture) {

            // set image to repeat instead of stretch
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
            const scale = 5;
            const aspect = viewport.offsetWidth / viewport.offsetHeight;
            texture.repeat.set(scale * aspect, scale);


            scene.background = texture;
        });


        camera = new THREE.PerspectiveCamera(75, viewport.offsetWidth / viewport.offsetHeight, 0.01, 1000);
        camera.position.x = 1;
        camera.position.y = 1;
        camera.position.z = 1;

        // Set up the WebGL renderer
        renderer = new THREE.WebGLRenderer();
        renderer.setSize(viewport.offsetWidth, viewport.offsetHeight);
        viewport.appendChild(renderer.domElement);

        // Add OrbitControls to allow interactive camera manipulation
        controls = new THREE.OrbitControls(camera, renderer.domElement);

        // Handle window resize
        window.addEventListener('resize', onWindowResize, false);

        // Start the animation loop
        animate();
    }

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    function onWindowResize() {
        camera.aspect = viewport.offsetWidth / viewport.offsetHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(viewport.offsetWidth, viewport.offsetHeight);

        updateColorbar();
    }

    function loadMesh(meshDataParam) {
        meshData = meshDataParam; // Store meshData globally to access color profiles later

        // Remove existing objects from the scene
        while (scene.children.length > 0) {
            scene.remove(scene.children[0]);
        }

        // Create geometry
        meshGeometry = new THREE.BufferGeometry();

        // Get the axis-aligned bounding box (AABB) of the mesh, and then center the mesh by translating AND scaling
        const aabb = new THREE.Box3();
        aabb.setFromPoints(meshData.vertices.map(v => new THREE.Vector3(...v)));
        const center = aabb.getCenter(new THREE.Vector3());
        const size = aabb.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 1 / maxDim;
        const offset = center.clone().multiplyScalar(-1);

        // Translate and scale vertices
        const verticesNormalized = meshData.vertices.map(v => {
            const vec = new THREE.Vector3(...v);
            vec.add(offset).multiplyScalar(scale);
            return vec.toArray();
        });

        // Set normalized vertices
        meshGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verticesNormalized.flat()), 3));


        // Vertices
        // const vertices = new Float32Array(meshData.vertices.flat());
        // meshGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

        // Faces
        const indices = new Uint32Array(meshData.faces.flat());
        meshGeometry.setIndex(new THREE.BufferAttribute(indices, 1));

        // Initial color profile (first one available)
        const initialProfileName = Object.keys(meshData.colorProfiles)[0];
        currentProfileName = initialProfileName;

        // Create color profile buttons and sliders
        createColorProfileControls();

        // Apply initial color profile
        applyColorProfile(initialProfileName);

        // Material with vertex colors
        meshMaterial = new THREE.MeshBasicMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            polygonOffset: true,
            polygonOffsetFactor: 1, // positive value pushes polygon further away
            polygonOffsetUnits: 1
        });

        // Create and add mesh
        mesh = new THREE.Mesh(meshGeometry, meshMaterial);
        scene.add(mesh);

        // Wireframe
        const wireframe = new THREE.WireframeGeometry(meshGeometry);
        const line = new THREE.LineSegments(wireframe);
        line.material.color.setHex(0x000000);
        line.material.opacity = 0.25;
        line.material.transparent = true;
        scene.add(line);

        // Axes
        addAxes();
    }

    function applyColorProfile(profileName, timeIndex = null) {
        currentProfileName = profileName;
        const profileData = meshData.colorProfiles[profileName];

        let scalarValues;
        let minScalar, maxScalar, range;
        if (Array.isArray(profileData)) {
            // Time-independent data
            scalarValues = profileData;
            // Use reduce to find min and max without exceeding call stack
            minScalar = scalarValues.reduce((a, b) => Math.min(a, b));
            maxScalar = scalarValues.reduce((a, b) => Math.max(a, b));
            range = maxScalar - minScalar || 1; // Avoid division by zero

            currentMinScalar = minScalar;
            currentMaxScalar = maxScalar;
        } else if (profileData && typeof profileData === 'object' && 'data' in profileData) {
            // Time-dependent data
            if (timeIndex === null) {
                timeIndex = 0;
            }
            currentTimeIndex = timeIndex;
            const dataArray = profileData.data;
            if (timeIndex < 0 || timeIndex >= dataArray.length) {
                console.error(`Time index ${timeIndex} out of range for profile "${profileName}".`);
                return;
            }
            scalarValues = dataArray[timeIndex];
            // find min and max across whole data array
            minScalar = dataArray.reduce((min, arr) => Math.min(min, arr.reduce((a, b) => Math.min(a, b))), Infinity);
            maxScalar = dataArray.reduce((max, arr) => Math.max(max, arr.reduce((a, b) => Math.max(a, b))), -Infinity);
            range = maxScalar - minScalar || 1; // Avoid division by zero

            currentMinScalar = minScalar;
            currentMaxScalar = maxScalar;
        } else {
            console.error(`Unknown data format for profile "${profileName}".`);
            return;
        }

        if (!scalarValues || scalarValues.length === 0) {
            console.error(`Scalar values for profile "${profileName}" are empty or undefined.`);
            return;
        }



        // Normalize the scalar values
        const normalizedScalars = scalarValues.map(val => (val - minScalar) / range);

        // Map normalized scalar values to colors using chroma.js
        const colors = new Float32Array(normalizedScalars.length * 3);  // 3 values per vertex (RGB)
        const scale = chroma.scale('viridis'); // Store scale for colorbar "viridis"
        colorScale = scale; // Save for later use

        normalizedScalars.forEach((scalar, i) => {
            const color = scale(scalar).rgb();  // Map scalar to 'viridis' color
            colors[i * 3] = color[0] / 255;      // Red
            colors[i * 3 + 1] = color[1] / 255;  // Green
            colors[i * 3 + 2] = color[2] / 255;  // Blue
        });

        // Update geometry colors
        meshGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        meshGeometry.attributes.color.needsUpdate = true;

        // Update colorbar
        updateColorbar();
    }

    function createColorProfileControls() {
        const colorProfilesContainer = document.getElementById('color-profiles');
        colorProfilesContainer.innerHTML = ''; // Clear previous controls

        const profileNames = Object.keys(meshData.colorProfiles);

        profileNames.forEach(profileName => {
            const profileData = meshData.colorProfiles[profileName];

            if (Array.isArray(profileData)) {
                // // Time-independent data, create button
                const button = document.createElement('button');
                button.className = 'color-profile-item';
                button.innerText = profileName;
                button.addEventListener('click', () => {
                    applyColorProfile(profileName);
                });
                colorProfilesContainer.appendChild(button);
            } else if (profileData && typeof profileData === 'object' && 'data' in profileData) {
                // Time-dependent data, create div with name and slider
                const container = document.createElement('div');
                container.className = 'color-profile-item';

                // Slider
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.min = 0;
                slider.max = profileData.data.length - 1;
                slider.value = 0;
                slider.step = 1;
                slider.addEventListener('input', () => {
                    applyColorProfile(profileName, parseInt(slider.value));
                });

                // Name label
                const nameLabel = document.createElement('p');
                nameLabel.innerText = profileName;
                nameLabel.style.cursor = 'pointer';
                nameLabel.addEventListener('click', () => {
                    applyColorProfile(profileName, parseInt(slider.value));
                });


                // Slider labels (start and end times)
                const startLabel = document.createElement('span');
                startLabel.innerText = profileData.start.toFixed(3);

                const endLabel = document.createElement('span');
                endLabel.innerText = profileData.end.toFixed(3);

                slider_container = document.createElement('div');
                slider_container.className = 'slider';
                slider_container.appendChild(startLabel);
                slider_container.appendChild(slider);
                slider_container.appendChild(endLabel);


                // Append elements
                container.appendChild(nameLabel);
                container.appendChild(slider_container);

                colorProfilesContainer.appendChild(container);
            } else {
                console.error(`Unknown data format for profile "${profileName}".`);
            }
        });
    }

    function addAxes() {
        const origin = new THREE.Vector3(0, 0, 0);
        const dirs = [
            new THREE.Vector3(1, 0, 0), // X-axis
            new THREE.Vector3(0, 1, 0), // Y-axis
            new THREE.Vector3(0, 0, 1), // Z-axis
        ];

        const length = 1;
        const arrowColors = [0xff0000, 0x00ff00, 0x0000ff];

        dirs.forEach((dir, i) => {
            const arrowHelper = new THREE.ArrowHelper(dir, origin, length, arrowColors[i]);
            // set render order to render on top of the mesh
            arrowHelper.renderOrder = -1;
            scene.add(arrowHelper);
        });
    }

    function fetchFileList() {
        const fileListContainer = document.getElementById('file-list');
        fetch('http://127.0.0.1:5000/json-files')
            .then(response => response.json())
            .then(files => {
                fileListContainer.innerHTML = ''; // Clear previous list

                files.forEach(file => {
                    const listItem = document.createElement('li');
                    listItem.className = 'file-list-item';
                    const link = document.createElement('a');
                    link.href = '#';
                    link.innerText = file;
                    link.addEventListener('click', e => {
                        e.preventDefault();
                        fetchFileContent(file).then(meshData => {
                            loadMesh(meshData);
                        });
                    });
                    listItem.appendChild(link);
                    fileListContainer.appendChild(listItem);
                });

                // Automatically load the first file if available
                if (files.length > 0) {
                    fetchFileContent(files[0]).then(meshData => {
                        loadMesh(meshData);
                    });
                }
            })
            .catch(error => console.error('Error fetching file list:', error));
    }

    function fetchFileContent(fileName) {
        return fetch(`http://127.0.0.1:5000/${fileName}`)
            .then(response => response.json())
            .catch(error => console.error('Error fetching file content:', error));
    }

    function updateColorbar() {
        const canvas = document.getElementById('colorbar');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Create gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        const numStops = 10;
        for (let i = 0; i <= numStops; i++) {
            const value = i / numStops;
            const color = colorScale(value).hex();
            gradient.addColorStop(1 - value, color); // Invert for top-to-bottom
        }

        // Fill rectangle with gradient
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);

        // Update labels
        document.getElementById('colorbar-min').innerText = currentMinScalar.toFixed(3);
        document.getElementById('colorbar-max').innerText = currentMaxScalar.toFixed(3);
    }

    // Initialize the app
    init();

    // Fetch the list of files when the page loads
    window.onload = fetchFileList;

})();

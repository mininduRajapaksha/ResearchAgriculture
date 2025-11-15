const video = document.getElementById('videoElement');
const stats = {
    fresh: document.getElementById('freshCount'),
    rotten: document.getElementById('rottenCount'),
    unripe: document.getElementById('unripeCount')
};

// Get camera access
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        console.error('Error accessing camera:', err);
    }
}

// Send frames to backend for processing
async function processFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Convert canvas to blob
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('frame', blob);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            // Update statistics
            const counts = { fresh: 0, rotten: 0, unripe: 0 };
            data.detections.forEach(det => {
                if (det.label in counts) {
                    counts[det.label]++;
                }
            });
            
            // Update display
            for (const [label, count] of Object.entries(counts)) {
                stats[label].textContent = count;
            }
            
        } catch (err) {
            console.error('Error processing frame:', err);
        }
    }, 'image/jpeg');
}

// Start camera when page loads
startCamera();

// Process frames every 100ms
setInterval(processFrame, 100);
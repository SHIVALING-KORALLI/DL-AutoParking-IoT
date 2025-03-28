// management.js
const socket = io();
const parkingSlots = document.getElementById('parkingSlots');
const detectedFrame = document.getElementById('detected-frame');
const noFeedMessage = document.querySelector('.no-feed-message');
const startDetectionButton = document.getElementById('startDetection');
const modal = document.getElementById('slotModal');
const modalContent = document.getElementById('modalContent');
const closeModal = document.querySelector('.close-modal');

let detectionActive = false;

// Modal function remains the same with plate number display
function openModal(slotDetails) {
    const timestamp = slotDetails.timestamp ? new Date(slotDetails.timestamp).toLocaleString() : 'N/A';
    const plateNumber = slotDetails.plate_number || 'N/A';
    
    modalContent.innerHTML = `
        <div class="modal-details">
            <p><strong>Slot ID:</strong> ${slotDetails.slotId}</p>
            <p><strong>Status:</strong> ${slotDetails.status}</p>
            <p><strong>Vehicle Type:</strong> ${slotDetails.vehicle_type || 'N/A'}</p>
            <p><strong>Number Plate:</strong> ${plateNumber}</p>
            <p><strong>Client Name:</strong> ${slotDetails.client_name || 'N/A'}</p>
            <p><strong>Last Updated:</strong> ${timestamp}</p>
        </div>
    `;
    
    modal.classList.add('show');
    document.body.style.overflow = 'hidden';
}

function closeModalFunction() {
    modal.classList.remove('show');
    document.body.style.overflow = 'auto';
}

// Event Listeners
closeModal.addEventListener('click', closeModalFunction);
modal.addEventListener('click', (e) => {
    if (e.target === modal) {
        closeModalFunction();
    }
});

startDetectionButton.addEventListener('click', () => {
    if (!detectionActive) {
        socket.emit('start_detection');
        detectionActive = true;
        startDetectionButton.textContent = 'Stop Detection';
    } else {
        socket.emit('stop_detection');
        detectionActive = false;
        startDetectionButton.textContent = 'Start Detection';
        detectedFrame.style.display = 'none';
        noFeedMessage.style.display = 'block';
    }
});

// Socket Events
socket.on('update_slots_and_frame', data => {
    if (data.frame) {
        detectedFrame.src = `data:image/jpeg;base64,${data.frame}`;
        detectedFrame.style.display = 'block';
        noFeedMessage.style.display = 'none';
    }    
    updateSlots(data.slots);
});

function updateSlots(slots) {
    parkingSlots.innerHTML = '';    
    
    Object.entries(slots).forEach(([slotId, details]) => {
        const slotElement = document.createElement('div');
        slotElement.className = `slot-card ${details.status.toLowerCase()}`;
        
        // Revert back to original slot card display without plate number
        slotElement.innerHTML = `
            <h3>${slotId}</h3>
            <div class="slot-status">${details.status}</div>
            <button class="view-more-btn">View More</button>
        `;
        
        const viewMoreBtn = slotElement.querySelector('.view-more-btn');
        viewMoreBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            openModal({
                slotId,
                ...details,
                plate_number: details.plate_number || 'N/A'  // Keep plate number for modal
            });
        });
        
        parkingSlots.appendChild(slotElement);
    });
}

// Rest of your existing socket event handlers
socket.on('detection_status', status => {
    if (status.active) {
        detectionActive = true;
        startDetectionButton.textContent = 'Stop Detection';
    } else {
        detectionActive = false;
        startDetectionButton.textContent = 'Start Detection';
        detectedFrame.style.display = 'none';
        noFeedMessage.style.display = 'block';
    }
});

socket.on('connect', () => {
    console.log('Connected to management dashboard');
    socket.emit('get_detection_status');
    socket.emit('get_slots');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    detectionActive = false;
    startDetectionButton.textContent = 'Start Detection';
    detectedFrame.style.display = 'none';
    noFeedMessage.style.display = 'block';
});

// Error handling
socket.on('error', (error) => {
    console.error('Socket error:', error);
    alert('An error occurred with the connection. Please refresh the page.');
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    socket.emit('get_slots');
    socket.emit('get_detection_status');
        // Add event listeners for the new buttons
    const manageUsersBtn = document.querySelector('.btn-manage-users');
    const viewHistoryBtn = document.querySelector('.btn-view-history');
        
        if (manageUsersBtn) {
            manageUsersBtn.addEventListener('click', () => {
                // Handle manage users button click
                window.location.href = '/manage-users'; // Or any other action
            });
        }
        
        if (viewHistoryBtn) {
            viewHistoryBtn.addEventListener('click', () => {
                // Handle view history button click
                window.location.href = '/view-history'; // Or any other action
            });
        }
});
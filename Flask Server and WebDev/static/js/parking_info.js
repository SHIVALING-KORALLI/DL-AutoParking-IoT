const socket = io();
let currentUser = '';
let availableSlots = [];
let selectedSlot = '';

function calculateDistanceFromEntrance(slotId) {
    const slotNum = parseInt(slotId.replace('slot', ''));
    if (isNaN(slotNum)) return Infinity;

    // Define rows and their entrance points
    const row1 = [1, 2, 3, 4];
    const row2 = [5, 6, 7, 8];
    
    // If slot 1 is free, it's automatically the nearest
    if (slotNum === 1) return 0;
    
    // Calculate distance within the row
    if (row1.includes(slotNum)) {
        return slotNum - 1; // Distance from slot 1
    } else if (row2.includes(slotNum)) {
        return slotNum - 5; // Distance from slot 5
    }
    
    return Infinity;
}

function findNearestFreeSlot(slots) {
    if (!slots) return null;
    
    const row1 = [1, 2, 3, 4];
    const row2 = [5, 6, 7, 8];
    
    // Get all free slots
    const freeSlots = Object.entries(slots)
        .filter(([_, slot]) => slot.status.toLowerCase() === 'free')
        .map(([_, slot]) => slot);
    
    if (freeSlots.length === 0) return null;
    
    // If slot 1 is free, it's automatically the nearest
    const slot1 = freeSlots.find(slot => parseInt(slot.slot_id.replace('slot', '')) === 1);
    if (slot1) return slot1;
    
    // Get free slots for each row
    const row1FreeSlots = freeSlots.filter(slot => 
        row1.includes(parseInt(slot.slot_id.replace('slot', ''))));
    const row2FreeSlots = freeSlots.filter(slot => 
        row2.includes(parseInt(slot.slot_id.replace('slot', ''))));
    
    // Calculate nearest free slot in each row
    const row1Nearest = row1FreeSlots.length > 0 ? 
        row1FreeSlots.reduce((a, b) => 
            calculateDistanceFromEntrance(a.slot_id) < calculateDistanceFromEntrance(b.slot_id) ? a : b) : null;
    
    const row2Nearest = row2FreeSlots.length > 0 ? 
        row2FreeSlots.reduce((a, b) => 
            calculateDistanceFromEntrance(a.slot_id) < calculateDistanceFromEntrance(b.slot_id) ? a : b) : null;
    
    // Compare distances between rows
    if (!row1Nearest) return row2Nearest;
    if (!row2Nearest) return row1Nearest;
    
    const row1Distance = calculateDistanceFromEntrance(row1Nearest.slot_id);
    const row2Distance = calculateDistanceFromEntrance(row2Nearest.slot_id);
    
    return row1Distance <= row2Distance ? row1Nearest : row2Nearest;
}

function getEnhancedSlotDescription(slot, slots) {
    if (!slot || !slots) return 'Slot information unavailable';

    const slotNum = parseInt(slot.slot_id.replace('slot', ''));
    const descriptions = {
        1: 'Close to entrance/exit, easy access',
        2: 'Wide space for large vehicles',
        3: 'Protected from weather',
        4: 'Corner spot, extra space',
        5: 'Direct entrance/exit access',
        6: 'Reserved for management users',
        7: 'Covered parking space',
        8: 'Quiet corner spot'
    };

    // For occupied slots
    if (slot.status.toLowerCase() === 'occupied') {
        return descriptions[slotNum];
    }

    // For free slots
    const nearestFree = findNearestFreeSlot(slots);
    const isNearest = nearestFree && nearestFree.slot_id === slot.slot_id;
    
    return `${descriptions[slotNum]}${isNearest ? '\nNearest to Entrance/Exit' : ''}`;
}

// Fix the createSlotCard function to only show release button for properly booked slots
function createSlotCard(slot, slots) {
    const slotNum = parseInt(slot.slot_id.replace('slot', ''));
    const card = document.createElement('div');
    card.className = `parking-slot ${slot.status.toLowerCase()}`;
    card.id = `slot${slotNum}`;
    
    // Set fixed dimensions to ensure consistent card sizes
    card.style.minHeight = '150px';
    card.style.width = '100%';
    
    const description = getEnhancedSlotDescription(slot, slots);
    const tooltipPosition = slotNum <= 4 ? 'top' : 'bottom';
    
    // FIXED: Only show release button if slot has both client_name AND timestamp (properly booked)
    // This ensures only slots that were actually booked by the user show a release button
    const isProperlyBooked = (
        slot.status.toLowerCase() === 'occupied' && 
        slot.client_name && 
        slot.client_name.toLowerCase() === currentUser.toLowerCase() &&
        slot.timestamp && 
        slot.timestamp !== '' &&
        typeof slot.booking_duration !== 'undefined'
    );
    
    // Simplified card content with tooltip
    card.innerHTML = `
        <div class="slot-header">
            <h3>Slot ${slotNum}</h3>
            <span class="status-badge ${slot.status.toLowerCase()}">
                ${slot.status}
            </span>
        </div>
        <div class="slot-tooltip ${tooltipPosition}">
            ${description}
        </div>
        <div class="slot-content" style="flex-grow: 1; display: flex; flex-direction: column; justify-content: flex-end;">
            ${isProperlyBooked ? `
                <div class="slot-actions" style="margin-top: auto;">
                    <button class="btn release-btn" type="button">Release Slot</button>
                </div>
            ` : ''}
        </div>
    `;

    // Add event listener for release button
    const releaseBtn = card.querySelector('.release-btn');
    if (releaseBtn) {
        releaseBtn.addEventListener('click', (e) => {
            e.preventDefault();
            releaseSlot(slot.slot_id); // FIXED: Pass the full slot.slot_id instead of just the number
        });
    }

    return card;
}


function updateSlots(slots) {
    const topRow = document.getElementById('topRow');
    const bottomRow = document.getElementById('bottomRow');
    const slotSelect = document.getElementById('slot-select');
    const bookingSection = document.getElementById('booking-section');
    
    // Clear existing slots but keep yield signs
    while (topRow.children.length > 1) {
        topRow.removeChild(topRow.lastChild);
    }
    while (bottomRow.children.length > 1) {
        bottomRow.removeChild(bottomRow.lastChild);
    }
    
    availableSlots = [];
    let occupiedSlotsCount = 0;

    if (!slots || Object.keys(slots).length === 0) {
        const message = document.createElement('p');
        message.className = 'no-slots-message';
        message.textContent = 'No parking slots available.';
        topRow.appendChild(message.cloneNode(true));
        bottomRow.appendChild(message.cloneNode(true));
    } else {
        Object.values(slots).forEach(slot => {
            // FIXED: Ensure the slot status is correct before rendering
            // This prevents showing both "free" status and release button
            if (slot.client_name && slot.client_name.toLowerCase() === currentUser.toLowerCase()) {
                // If the user owns this slot, ensure it shows as occupied
                slot.status = 'Occupied';
            }
            
            const slotCard = createSlotCard(slot, slots);
            const slotNum = parseInt(slot.slot_id.replace('slot', ''));
            
            if (slotNum <= 4) {
                topRow.appendChild(slotCard);
            } else {
                bottomRow.appendChild(slotCard);
            }

            if (slot.status.toLowerCase() === 'free') {
                // Only add non-reserved slots (slot 6 is reserved)
                if (slotNum !== 6) {
                    availableSlots.push(slot.slot_id);
                }
            } else if (slot.status.toLowerCase() === 'occupied') {
                occupiedSlotsCount++;
            }
        });
    }

    // Update slot selection dropdown
    if (slotSelect) {
        const currentSelection = slotSelect.value;
        slotSelect.innerHTML = '<option value="">Select a Free Slot</option>';
        availableSlots.forEach(slotId => {
            const option = document.createElement('option');
            option.value = slotId;
            const slotNum = parseInt(slotId.replace('slot', ''));
            const nearestFree = findNearestFreeSlot(slots);
            option.textContent = `Slot ${slotNum} ${nearestFree && nearestFree.slot_id === slotId ? '(Nearest to entrance)' : ''}`;
            if (slotId === currentSelection || slotId === selectedSlot) {
                option.selected = true;
                selectedSlot = slotId;
            }
            slotSelect.appendChild(option);
        });

        // Handle booking section visibility
        if (bookingSection) {
            // Disable booking if 5 or more slots are occupied (2 or fewer slots free)
            if (occupiedSlotsCount >= 5) {
                bookingSection.classList.add('disabled');
                showNotification('Booking is currently unavailable due to capacity limits', 'warning');
            } else {
                bookingSection.classList.remove('disabled');
            }
        }
    }
}

function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    if (!notification) {
        console.error('Notification element not found');
        return;
    }

    // Clear any existing timeouts
    if (notification.timeout) {
        clearTimeout(notification.timeout);
    }

    // Add animation classes
    notification.className = 'notification';
    // Force a reflow
    void notification.offsetWidth;
    notification.className = `notification ${type} show`;
    
    // Set the message
    notification.textContent = message;

    // Auto-hide after 5 seconds
    notification.timeout = setTimeout(() => {
        notification.className = 'notification';
    }, 5000);

    // Log notification for debugging
    console.log(`Notification shown: ${message} (${type})`);
}

function bookSlot() {
    const slotSelect = document.getElementById('slot-select');
    const timeSelect = document.getElementById('time-select');
    const vehicleSelect = document.getElementById('vehicle-type-select');
    
    const slotId = slotSelect.value;
    // Make sure we're getting and sending an integer in seconds
    const bookingDuration = parseInt(timeSelect.value);
    const vehicleType = vehicleSelect.value;

    // Add validation and logging
    if (!slotId) {
        showNotification('Please select a slot', 'error');
        return;
    }

    if (isNaN(bookingDuration)) {
        showNotification('Invalid booking duration', 'error');
        return;
    }
    
    console.log(`Booking slot ${slotId} for ${bookingDuration} seconds (${bookingDuration/60} minutes)`);

    // Verify slot is still free before booking
    if (!availableSlots.includes(slotId)) {
        showNotification('Selected slot is no longer available', 'error');
        selectedSlot = ''; // Only reset if the slot is no longer available
        socket.emit('get_slots');
        return;
    }

    // Disable the book button temporarily to prevent double-clicking
    const bookButton = document.getElementById('book-slot-btn');
    bookButton.disabled = true;
    bookButton.textContent = 'Booking...';

    // Make sure we're sending the booking duration as an integer
    const bookingData = {
        spot_id: slotId,
        vehicle_type: vehicleType,
        booking_duration: bookingDuration
    };
    
    console.log('Sending booking data:', bookingData);
    
    socket.emit('book_spot', bookingData, response => {
        // Re-enable the book button
        bookButton.disabled = false;
        bookButton.textContent = 'Book Slot';
        
        console.log('Booking response:', response);
        
        if (response.success) {
            showNotification(response.message, 'success');
            selectedSlot = ''; // Only reset after successful booking
        } else {
            showNotification(response.message, 'error');
            // Keep the selected slot in case of booking failure
        }
        socket.emit('get_slots');
    });
}

// Fix the releaseSlot function to pass the complete slot ID
function releaseSlot(slotId) {
    // Ensure slotId is correctly formatted
    if (!slotId.startsWith('slot')) {
        slotId = `slot${slotId}`;
    }
    
    console.log(`Attempting to release slot: ${slotId}`);
    
    // Add loading state to prevent double-clicks
    const releaseBtn = document.querySelector(`#${slotId} .release-btn`);
    if (releaseBtn) {
        releaseBtn.disabled = true;
        releaseBtn.textContent = 'Releasing...';
    }

    socket.emit('release_spot', { spot_id: slotId }, response => {
        console.log(`Release response:`, response);
        
        if (releaseBtn) {
            releaseBtn.disabled = false;
            releaseBtn.textContent = 'Release Slot';
        }

        if (response.success) {
            showNotification(response.message, 'success');
        } else {
            showNotification(response.message || 'Failed to release slot', 'error');
        }
        // Force refresh the slots immediately
        socket.emit('get_slots');
    });
}

// Socket event handlers
socket.on('connect', () => {
    console.log('Connected to server');
    fetch('/get_current_user')
        .then(response => response.json())
        .then(data => {
            if (data && data.name) {
                currentUser = data.name;
                socket.emit('get_slots');
            } else {
                console.error('Failed to get user name from response');
            }
        })
        .catch(error => {
            console.error('Error fetching current user:', error);
            showNotification('Failed to get user information', 'error');
        });
});

socket.on('update_slots_client', data => {
    updateSlots(data.slots);
});

socket.on('slot_status_changed', data => {
    // When slot status changes, immediately refresh all slots
    updateSlots(data.slots);
});

socket.on('slot_auto_released', data => {
    showNotification(`Slot ${data.slot_id} has been automatically released`);
    // Force refresh slots when auto-released
    socket.emit('get_slots');
});

socket.on('twilio_notification_sent', data => {
    showNotification(`SMS notification sent to ${data.recipient}`, 'success');
});

socket.on('twilio_notification_failed', data => {
    showNotification(`Failed to send SMS notification: ${data.error}`, 'error');
});

socket.on('slot_booking_failed', data => {
    showNotification(data.message || 'Booking failed', 'error');
    // Reset the book button if booking failed
    const bookButton = document.getElementById('book-slot-btn');
    if (bookButton) {
        bookButton.disabled = false;
        bookButton.textContent = 'Book Slot';
    }
    socket.emit('get_slots');
});

socket.on('slot_release_failed', data => {
    showNotification(data.message || 'Release failed', 'error');
    socket.emit('get_slots');
});

socket.on('disconnect', () => {
    showNotification('Lost connection to server. Attempting to reconnect...', 'error');
});

socket.on('reconnect', () => {
    showNotification('Reconnected to server', 'success');
    socket.emit('get_slots');
});

// Event listeners for booking section
document.addEventListener('DOMContentLoaded', () => {
    const bookButton = document.getElementById('book-slot-btn');
    bookButton.addEventListener('click', bookSlot);

    // Additional event listener to update selectedSlot when manually changed
    const slotSelect = document.getElementById('slot-select');
    slotSelect.addEventListener('change', (event) => {
        selectedSlot = event.target.value;
    });
    
    // Initial fetch of slots
    socket.emit('get_slots');
    
    // Set up periodic refresh to ensure slots are in sync
    setInterval(() => {
        socket.emit('get_slots');
    }, 10000); // Refresh every 10 seconds
});
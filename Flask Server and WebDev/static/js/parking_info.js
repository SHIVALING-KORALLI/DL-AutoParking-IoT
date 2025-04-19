// First, let's consolidate the duplicate slot_auto_released event handlers
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
        .map(([slotId, slot]) => ({ ...slot, slot_id: slotId }));
    
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

function createSlotCard(slot, slots) {
    const slotNum = parseInt(slot.slot_id.replace('slot', ''));
    const card = document.createElement('div');
    
    // Handle special case for slot 6 - It should always show as reserved unless occupied
    let statusClass = slot.status.toLowerCase();
    if (slotNum === 6 && statusClass === 'free') {
        statusClass = 'reserved';
    }
    
    card.className = `parking-slot ${statusClass}`;
    card.id = `${slot.slot_id}`; // Ensure we use the actual slot_id from the data
    
    // Set fixed dimensions to ensure consistent card sizes
    card.style.minHeight = '150px';
    card.style.width = '100%';
    
    const description = getEnhancedSlotDescription(slot, slots);
    const tooltipPosition = slotNum <= 4 ? 'top' : 'bottom';
    
    // Check if this slot belongs to the current user
    const isUsersSlot = (
        slot.status.toLowerCase() === 'occupied' && 
        slot.client_name && 
        slot.client_name.toLowerCase() === currentUser.toLowerCase()
    );
    
    // Add reserved message for slot 6
    const reservedTooltip = slotNum === 6 && statusClass === 'reserved' ? 
        `<div class="slot-reserved-tooltip">Reserved for Management</div>` : '';
    
    // Simplified card content with tooltip - REMOVED client_name display
    card.innerHTML = `
        ${reservedTooltip}
        <div class="slot-header">
            <h3>Slot ${slotNum}</h3>
            <span class="status-badge ${statusClass}">
                ${slotNum === 6 && statusClass === 'reserved' ? 'Reserved' : slot.status}
            </span>
        </div>
        <div class="slot-tooltip ${tooltipPosition}">
            ${description}
        </div>
        <div class="slot-content" style="flex-grow: 1; display: flex; flex-direction: column; justify-content: flex-end;">
            ${isUsersSlot ? `
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
            releaseSlot(slot.slot_id);
        });
    }

    return card;
}

function updateSlots(slots) {
    console.log('Updating slots with:', slots);
    const topRow = document.getElementById('topRow');
    const bottomRow = document.getElementById('bottomRow');
    const slotSelect = document.getElementById('slot-select');
    const bookingSection = document.getElementById('booking-section');
    
    if (!topRow || !bottomRow) {
        console.error('Could not find topRow or bottomRow elements');
        return;
    }
    
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
        Object.entries(slots).forEach(([slotId, slotData]) => {
            // Make sure the slot has all required properties
            const slot = {
                ...slotData,
                slot_id: slotId // Ensure slot_id is included in the data
            };
            
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

// Improved socket connection handling with retry mechanism
socket.on('connect', () => {
    console.log('Connected to server');
    
    // Clear any reconnection notification
    const reconnectNotification = document.getElementById('reconnect-notification');
    if (reconnectNotification) {
        reconnectNotification.style.display = 'none';
    }
    
    // Get current user and then fetch slots
    fetch('/get_current_user')
        .then(response => response.json())
        .then(data => {
            if (data && data.name) {
                currentUser = data.name;
                console.log(`Current user: ${currentUser}`);
                
                // Request initial slots data
                socket.emit('get_slots');
            } else {
                console.error('Failed to get user name from response');
                showNotification('Failed to get user information', 'error');
            }
        })
        .catch(error => {
            console.error('Error fetching current user:', error);
            showNotification('Failed to get user information', 'error');
        });
});

// FIXED: Consolidated slot_auto_released event handler
socket.on('slot_auto_released', data => {
    console.log('Slot auto-released received:', data);
    const slotNum = data.slot_id.replace('slot', '');
    showNotification(`Slot ${slotNum} has been automatically released`, 'info');
    
    // Force refresh slots immediately
    console.log('Requesting slots update after auto-release');
    socket.emit('get_slots');
    
    // Also update the UI directly to ensure immediate visual feedback
    const slotElement = document.getElementById(data.slot_id);
    if (slotElement) {
        slotElement.className = 'parking-slot free';
        const statusBadge = slotElement.querySelector('.status-badge');
        if (statusBadge) {
            statusBadge.className = 'status-badge free';
            statusBadge.textContent = 'Free';
        }
        
        // Remove any release buttons
        const releaseBtn = slotElement.querySelector('.release-btn');
        if (releaseBtn) {
            const actions = releaseBtn.closest('.slot-actions');
            if (actions) {
                actions.remove();
            }
        }
        
        console.log(`Updated UI for auto-released slot ${data.slot_id}`);
    }
});

// Improved update_slots_client handler with better error handling
socket.on('update_slots_client', data => {
    console.log('Received slot update:', data);
    
    // Check if we actually got slots data
    if (!data || !data.slots) {
        console.error('Invalid slots data received:', data);
        return;
    }
    
    // Log the current status of each slot
    Object.entries(data.slots).forEach(([slotId, slot]) => {
        console.log(`${slotId}: status=${slot.status}, client=${slot.client_name || 'none'}`);
    });
    
    // Update the UI with the new slots data
    updateSlots(data.slots);
});

// FIXED: Add event handler for update_slots_and_frame
socket.on('update_slots_and_frame', data => {
    console.log('Received slots and frame update');
    if (data && data.slots) {
        updateSlots(data.slots);
    }
});

socket.on('slot_status_changed', data => {
    // When slot status changes, immediately refresh all slots
    console.log('Slot status changed:', data);
    if (data && data.slots) {
        updateSlots(data.slots);
    } else {
        // If no slots data provided, request fresh data
        socket.emit('get_slots');
    }
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
    console.error('Disconnected from server');
    showNotification('Lost connection to server. Attempting to reconnect...', 'error');
    
    // Show reconnection message
    const reconnectNotification = document.getElementById('reconnect-notification');
    if (reconnectNotification) {
        reconnectNotification.style.display = 'block';
    }
});

socket.on('reconnect', () => {
    console.log('Reconnected to server');
    showNotification('Reconnected to server', 'success');
    
    // Hide reconnection message
    const reconnectNotification = document.getElementById('reconnect-notification');
    if (reconnectNotification) {
        reconnectNotification.style.display = 'none';
    }
    
    // Request fresh data
    socket.emit('get_slots');
});

// Add code to inject a reconnection notification at the top of the page
document.addEventListener('DOMContentLoaded', () => {
    // Create reconnection notification element
    const reconnectNotification = document.createElement('div');
    reconnectNotification.id = 'reconnect-notification';
    reconnectNotification.className = 'notification error';
    reconnectNotification.textContent = 'Lost connection to server. Attempting to reconnect...';
    reconnectNotification.style.display = 'none';
    reconnectNotification.style.position = 'fixed';
    reconnectNotification.style.top = '10px';
    reconnectNotification.style.left = '50%';
    reconnectNotification.style.transform = 'translateX(-50%)';
    reconnectNotification.style.zIndex = '9999';
    
    // Add to body
    document.body.appendChild(reconnectNotification);
    
    // Set up button event listeners
    const bookButton = document.getElementById('book-slot-btn');
    if (bookButton) {
        bookButton.addEventListener('click', bookSlot);
    }

    // Additional event listener to update selectedSlot when manually changed
    const slotSelect = document.getElementById('slot-select');
    if (slotSelect) {
        slotSelect.addEventListener('change', (event) => {
            selectedSlot = event.target.value;
        });
    }
    
    // Initial fetch of slots
    socket.emit('get_slots');
    
    // Create a notification container if it doesn't exist
    if (!document.getElementById('notification')) {
        const notificationElement = document.createElement('div');
        notificationElement.id = 'notification';
        notificationElement.className = 'notification';
        document.body.appendChild(notificationElement);
    }
    
    // Set up periodic refresh to ensure slots are in sync
    setInterval(() => {
        if (socket.connected) {
            console.log('Periodic refresh: requesting latest slot data');
            socket.emit('get_slots');
        }
    }, 5000); // Refresh every 5 seconds for better responsiveness
});
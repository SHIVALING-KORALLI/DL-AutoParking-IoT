document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const errorMessage = document.querySelector('.error-message');    
    if (!loginForm || !errorMessage) {
        console.error('Required elements not found');
        return;
    }
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();     
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const role = document.getElementById('role').value;
        // Show loading state
        const submitButton = loginForm.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = 'Logging in...';
        }
        // Clear previous error
        errorMessage.style.display = 'none';
        errorMessage.textContent = '';

        fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password, role })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Redirect based on role
                if (data.role === 'management') {
                    window.location.href = '/management';
                } else {
                    window.location.href = '/parking_info';
                }
            } else {
                errorMessage.textContent = data.message || 'Login failed. Please check your credentials.';
                errorMessage.style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Login error:', error);
            errorMessage.textContent = 'Network error. Please try again.';
            errorMessage.style.display = 'block';
        })
        .finally(() => {
            // Reset button state
            if (submitButton) {
                submitButton.disabled = false;
                submitButton.textContent = 'Login';
            }
        });
    });
});

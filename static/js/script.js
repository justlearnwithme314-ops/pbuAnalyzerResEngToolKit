document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
    const submitBtn = document.querySelector('button[type="submit"]');

    // Add loading state to the submit button
    if (form && submitBtn) {
        form.addEventListener('submit', () => {
            // Change button text and appearance
            submitBtn.textContent = 'Processing... Please wait';
            submitBtn.disabled = true;
            
            // Optional: You could also add logic here to check if a file 
            // is actually selected before allowing the submission, 
            // depending on your backend requirements.
        });
    }
});
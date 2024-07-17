function selectDataset(dataset) {
    // Update the selected dataset
    selectedDataset = dataset;
    // // Update the button label
    // document.getElementById('dataset-button').textContent = dataset;
    document.getElementById('selected-dataset').value = selectedDataset; // Update hidden input field

    // Highlight the selected dataset
    updateDatasetHighlighting(dataset);
}

function updateDatasetHighlighting(selectedDataset) {
    // Remove highlighting from all dataset options
    document.querySelectorAll('.dropdown-content a').forEach(function(link) {
        link.classList.remove('selected-object');
    });

    // Add highlighting to the selected dataset
    let selectedLink = document.querySelector('.dropdown-content a[onclick*="' + selectedDataset + '"]');
    if (selectedLink) {
        selectedLink.classList.add('selected-object');
    }
}

document.getElementById('search-form').addEventListener('submit', function() {
    // document.getElementById('loading-icon').style.display = 'block';
    var inputField = document.getElementById('search-box');
    if (!inputField.value.trim()) {
        // Prevent the form from submitting
        event.preventDefault();
        // // Optionally, you can alert the user or handle the empty input case in another way here
        // alert('Please enter a search term.');
    } else {
        // If the input field is not empty, display the loading icon
        document.getElementById('loading-icon').style.display = 'block';
    }

});


// Initialize highlighting on page load
document.addEventListener('DOMContentLoaded', function() {
    updateDatasetHighlighting(selectedDataset);
});

function updatePopupContent(newContent) {
    document.getElementById('popupText').innerHTML = newContent;
}

document.getElementById('helpButton').onclick = function() {
    updatePopupContent("<h2>Welcome to CLIP-Branches</h2>" + 
    "<p>If you need help on how to use the search engine, please refer to the tutorial video below.</p>" +
    "<p><strong>Tutorial-Video:</strong> <a href='https://youtu.be/lepPM3zi0l8' target='_blank'>https://youtu.be/lepPM3zi0l8</a></p>" +
    "<p>If you want to setup CLIP-Branches yourself for your own data or have further questions, please refer to our Github-Repository.</p>" +
    "<p><strong>Github Repository:</strong> <a href='https://github.com/cluel01/clip-branches' target='_blank'>https://github.com/cluel01/clip-branches</a></p>"
 );
    document.getElementById('popupBox').style.display = "block";
}

document.getElementsByClassName('close')[0].onclick = function() {
    document.getElementById('popupBox').style.display = "none";
}

window.onclick = function(event) {
    if (event.target == document.getElementById('popupBox')) {
        document.getElementById('popupBox').style.display = "none";
    }
}

if (demo == "true") {
    document.getElementById('helpButton').click();
}

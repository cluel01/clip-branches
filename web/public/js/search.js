positiveIndices = [];
negativeIndices = [];
currentIndex = 0;
batchSize = 15;
res_data = null;
sql_statements = null;

const positiveColor = 'green';
const negativeColor = 'red';

document.getElementById('finetune-search-button').addEventListener('click', () => {
    sendFineTuneSearchRequest(positiveIndices, negativeIndices,selectedDataset,selectedSearcher);
    //href to results page
});


function initImagePatches() {
    // ADD all unselected images to positiveIndices
    var image_frames = document.getElementsByClassName('image-frame');
    for (var i = 0; i < image_frames.length; i++) {
        iid = image_frames[i].children[0].getAttribute('idx');
        positiveIndices.push(parseInt(iid));
    }
    markImagesSearch();

}

function markAllImages(type) {
    if (type === 'positive') {
        color = positiveColor;
    }
    else if (type === 'negative') {
        color = negativeColor;
    }
    else {
        color = 'transparent';
    }

    var image_frames = document.getElementsByClassName('image-frame');
    for (var i = 0; i < image_frames.length; i++) {
        iid = image_frames[i].children[0].getAttribute('idx');
        currentBorderColor = image_frames[i].style.borderColor;
        
        if (currentBorderColor != color) {
            image_frames[i].style.borderColor = color;
            if (type === 'positive') {
                positiveIndices.push(parseInt(iid));
                remove_element(parseInt(iid),'negative');
            }
            else if (type === 'negative') {
                negativeIndices.push(parseInt(iid));
                remove_element(parseInt(iid),'positive');
            }
            else {
                remove_element(parseInt(iid),'positive');
                remove_element(parseInt(iid),'negative');
            }
        }
    }
}

async function sendFineTuneSearchRequest(positiveIndices, negativeIndices,selectedDataset,selectedSearcher) {
    numNegSamples = document.getElementById("negative-slider").value;
    negativeWeight = document.getElementById("negative-weight-slider").value;
    sql_statements = null;
    fetch("/finetune-search", {
        method: "POST",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({positive: positiveIndices, negative: negativeIndices,dataset:selectedDataset,searcher:selectedSearcher,numNegSamples:numNegSamples,negativeWeight:negativeWeight})
    }).then(response => response.json()).then(data => {
        sql_statements = data.sql_statements;
        currentIndex = 0;
        return updateResults(data);
    }).then(() => {
        markImagesSearch()}).catch(error => console.error('Error:', error));
}


function markImagesSearch(startIdx = 0) {
    console.log("marking images");
    // var image_frames = document.getElementsByClassName('image-frame');
    const grid = document.querySelector('.images-grid');
    for (var i = startIdx; i < grid.children.length; i++) {
        if (grid.children[i].className === 'image-frame') {
            var image_frame = grid.children[i];
            iid = image_frame.children[0].getAttribute('idx');
            if (positiveIndices.includes(parseInt(iid))) {
                image_frame.style.borderColor = positiveColor;
            }
            if (negativeIndices.includes(parseInt(iid))) {
                image_frame.style.borderColor = negativeColor;
            }
        }
    }
    // for (var i = 0; i < image_frames.length; i++) {
    //     iid = image_frames[i].children[0].getAttribute('idx');
    //     if (positiveIndices.includes(parseInt(iid))) {
    //         image_frames[i].style.borderColor = 'red';
    //     }
    //     if (negativeIndices.includes(parseInt(iid))) {
    //         image_frames[i].style.borderColor = 'blue';
    //     }
    // }
}

function updateResults(data) {
    return new Promise(resolve => {
        const grid = document.querySelector('.images-grid');
        grid.innerHTML = ''; // Clear existing images

        document.getElementById('query-time-value').textContent = data.queryTime;
        document.getElementById('num-results-value').textContent = data.nresults;

        for (let i = 0; i < Math.min(batchSize, data.images.length); i++) {
            const imageFrame = createImageFrame(data.images[i], i);
            imageFrame.addEventListener('dblclick', (e) => {
                e.preventDefault();
            });
            grid.appendChild(imageFrame);
            currentIndex++;
        }

        console.log(data.images.length);
        console.log(data)
    
        // Show the 'Load More' button if there are more images to load
        if (currentIndex < data.images.length) {
            document.getElementById('load-more-button').style.display = 'block';
        }

        res_data = data;

        resolve();

    });
}

function createImageFrame(img, index) {
    const imageFrame = document.createElement('div');
    imageFrame.className = 'image-frame';
    imageFrame.id = 'image-frame-' + index;

    const image = document.createElement('img');
    image.src = img[1];
    image.alt = 'Result Image ' + (index + 1);
    image.setAttribute('idx', img[0]);
    image.className = 'result-image';
    image.title = img[0];
    image.addEventListener('click', (e) => {
        e.preventDefault();
    });
    image.addEventListener('dblclick', (e) => {
        e.preventDefault();
    });
    image.onclick = function() { markImage('image-frame-' + index); };
    imageFrame.appendChild(image);
    return imageFrame;
}


function markImage(imageId) {
    var imageFrame = document.getElementById(imageId);
    var imageIdx = parseInt(imageFrame.getElementsByClassName("result-image")[0].getAttribute("idx"));
    var currentBorderColor = imageFrame.style.borderColor;
    // if (!(event.button === 0 && firstSearch)) {
    //     // Determine the new border color based on the mouse button clicked
    //     if (event.button === 0) { // Left click for red
    //         newBorderColor = positiveColor;
    //         type = 'positive';
    //     } else if (event.button === 2) { // Right click for blue
    //         newBorderColor = negativeColor;
    //         type = 'negative';
    //     }

    // Toggle the border color: if it's already the new color, remove it, otherwise set it
    if (currentBorderColor === positiveColor) {
        imageFrame.style.borderColor = negativeColor; // or '' for no border
        remove_element(imageIdx,"positive");
        negativeIndices.push(imageIdx);
    } 
    else if (currentBorderColor === negativeColor) {
        imageFrame.style.borderColor = "transparent";
        remove_element(imageIdx,"negative");
    }
    else {
        imageFrame.style.borderColor = positiveColor;
        positiveIndices.push(imageIdx);
    }

        // // Prevent default behavior for right-click
        // if (event.preventDefault) event.preventDefault();
        // if (event.stopPropagation) event.stopPropagation();
    
    
}

function remove_element(id,type) {
    var list;
    if (type === 'positive') {
        list = positiveIndices;
    } else {
        list = negativeIndices;
    }

    var index = list.indexOf(id);
    if (index > -1) { 
    list.splice(index, 1); 
    }
}

// // Disable the context menu on the entire document
// document.addEventListener('contextmenu', function(event) {
//     event.preventDefault();
// }, false);

// // Attach the markImage function to each image frame
// window.onload = function() {
//     var imageFrames = document.getElementsByClassName('image-frame');
//     for (var i = 0; i < imageFrames.length; i++) {
//         imageFrames[i].addEventListener('click', function(event) {
//             markImage(event, this.id);
//         });
//         imageFrames[i].addEventListener('contextmenu', function(event) {
//             markImage(event, this.id);
//         });
//     }
// };

document.addEventListener('DOMContentLoaded', function() {
    var imagesGrid = document.querySelector('.images-grid');
    if (imagesGrid) {
        imagesGrid.addEventListener('click', function(event) {
            var target = event.target;
            if (target.className === 'result-image') {
                markImage(target.parentElement.id);
            }
        });

    }
});

document.getElementById('load-more-button').addEventListener('click', function() {
    const grid = document.querySelector('.images-grid');

    const start = currentIndex;
    const end = Math.min(currentIndex + batchSize, res_data.images.length);
    for (let i = start; i < end; i++) {
        const imageFrame = createImageFrame(res_data.images[i], i);
        grid.appendChild(imageFrame);
        currentIndex++;
    }

    markImagesSearch(start);

    // Hide the button if all images are loaded
    if (currentIndex >= res_data.images.length) {
        document.getElementById('load-more-button').style.display = 'none';
    }
});



function selectSearcher(searcher) {
    console.log(searcher);
    // Update the selected dataset
    selectedSearcher = searcher;

    // document.getElementById('selected-searcher').value = searcher; 

    // Highlight the selected dataset
    updateSearcherHighlighting(searcher);
}

function updateSearcherHighlighting(selectedSearcher) {
    // Remove highlighting from all dataset options
    document.querySelectorAll('.dropdown-content a').forEach(function(link) {
        link.classList.remove('selected-object');
    });

    // Add highlighting to the selected dataset
    let selectedLink = document.querySelector('.dropdown-content a[onclick*="' + selectedSearcher + '"]');
    if (selectedLink) {
        selectedLink.classList.add('selected-object');
    }
}



// Initialize highlighting on page load
document.addEventListener('DOMContentLoaded', function() {
    updateSearcherHighlighting(selectedSearcher);
});

function reset() {
    positiveIndices = [];
    negativeIndices = [];
}


function updatePopupContent(newContent) {
    document.getElementById('popupText').innerHTML = newContent;
}

document.getElementById('sqlButton').onclick = function() {
    if (sql_statements === null || sql_statements.length === 0) {
        updatePopupContent("No SQL statements generated! Use Decision Branches or BoxNet models for fine-tuning instead!")
    }
    else {
        updatePopupContent(sql_statements[0])
    }
    
    document.getElementById('popupBox').style.display = "block";
}

document.getElementById('helpButton').onclick = function() {
    updatePopupContent("<h2>Manual:</h2>" + 
                        "<p><strong>Initial Search Results:</strong> The images displayed are the top results from the initial text search. Click on an image to mark it as positive (green) or negative (red).</p>" +
                       "<p><strong>Fine-tune Search:</strong> Click the 'Fine-tune Search' button to refine your results. Use the 'lens' button to select the model for fine-tuning.</p>" +
                       "<p><strong>Adjusting Negative Samples:</strong> Use the slider to adjust the number of negative random samples added to your search. More negative samples can lead to more precise results but may increase training times. Adjust according to your requirements.</p>" +
                       "<p>You can fine-tune the results in multiple iterations to meet your specific needs.</p>");
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


// update Slider values
function updateNumResults(sliderType,value) {
    if (sliderType === 'negative') {
        document.getElementById("negative-slider-value").textContent = value;
    }
    else if (sliderType === 'weight') {
        document.getElementById("negative-weight-slider-value").textContent = value;
    }
}

// start
initImagePatches();
if (demo == "true") {
    document.getElementById('helpButton').click();
}

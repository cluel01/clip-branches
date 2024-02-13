const express = require('express');
const app = express();
const path = require('path');
const axios = require('axios');
const bodyParser = require('body-parser');


const dOptions = {"LAION-262M":"laion_262m_var","CIFAR10":"cifar","Shutterstock":"shutterstock"}

const sOptions = {"DBranches":"db","DBranches Ensemble":"ens","Decision Tree":"dtree","Random Forest":"rf"}

const data_size = 150
const maxTextResults = 60;
const maxNegSamples = 100000;
const negSamples = 30000;
const maxImages = 1000
const endpoint_data = `%ENDPOINT_PROTOCOL%://%ENDPOINT_DATA%/image/`
const demo = `%DEMO%`
const negativeWeight = 1

// Create an Axios instance with proxy disabled
const axiosInstance = axios.create({
    proxy: false
});


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));


// Set the view engine to ejs
app.set('view engine', 'ejs');

// Set the folder for static files (like CSS)
app.use(express.static(path.join(__dirname, 'public')));

// Route for the home page
app.get('/', (req, res) => {
    const datasetOptions = Object.keys(dOptions);//dOptions; // Array of dataset names
    const defaultDataset = datasetOptions[0]; // Default dataset name
    

    res.render('index',{ datasetOptions: datasetOptions, defaultDataset: defaultDataset,demo:demo });
});

app.get('/search', async (req, res) => {
    const searchOptions = Object.keys(sOptions);//sOptions; 
    const defaultSearcher = searchOptions[0];
    const searchQuery = req.query.query;
    const dataset_key = req.query.dataset;
    dataset = dOptions[dataset_key];

    try {
        // Assuming you receive an array of image indexes from the first API
        const results = await fetchSearchserviceTextSearch(searchQuery,dataset,maxTextResults); // Replace with your actual API call
        const imageIndexes = results.results; 
        const time = results.time.toFixed(3);

        const images = imageIndexes.map(idx => 
            [idx,endpoint_data+dataset+"/"+idx+`?size=${data_size}`]
        );
        
        console.log(images)
        // Extract the URLs from the responses
        // const images = imageUrls.map(response => response.data.url); // Adjust based on actual response structure

        // Render the results page with the image URLs
        res.render('results', { images: images,searchDataset:dataset,queryTime:time,searchOptions:searchOptions,defaultSearcher:defaultSearcher,
            maxNegSamples:maxNegSamples,negSamples:negSamples,demo:demo,negativeWeight:negativeWeight });
    } catch (error) {
        console.error('API error:', error);
        res.render('error', { message: 'Error fetching images' });
    }
});

app.post('/finetune-search', async (req, res) => {
    const positiveIndices = req.body.positive;
    const negativeIndices = req.body.negative;
    const dataset = req.body.dataset;
    const searcher_key = req.body.searcher;
    const searcher = sOptions[searcher_key];
    const numNegSamples = req.body.numNegSamples;
    const negativeWeight = req.body.negativeWeight;
    
    console.log("Finetune:",positiveIndices);
    console.log("Finetune: ",negativeIndices);

    try {
        // Assuming you receive an array of image indexes from the first API
        const results = await fetchSearchserviceFinesearch(positiveIndices,negativeIndices,dataset,searcher,numNegSamples,negativeWeight); // Replace with your actual API call
        console.log(results)
        const imageIndexes = results.results; 
        const time = results.time.toFixed(3);;
        const nresults = results.nresults;
        const sql_statements = results.sql_statements;

        
        const images = imageIndexes.map(idx => 
            [idx,endpoint_data+dataset+"/"+idx+`?size=${data_size}`]
        );
        
        // Extract the URLs from the responses
        // const images = imageUrls.map(response => response.data.url); // Adjust based on actual response structure

        // // Render the results page with the image URLs
        // res.render('results', { images: images });
        res.json({images: images,queryTime:time,nresults:nresults,sql_statements:sql_statements})
    } catch (error) {
        console.error('API error:', error);
        res.render('error', { message: 'Error fetching images' });
    }
});



async function fetchSearchserviceTextSearch(searchText) {
    let endpoint = `%ENDPOINT_PROTOCOL%://%ENDPOINT_SEARCHSERVICE%/search_text/${dataset}/${searchText}`
    console.log(endpoint)
    const response = await axiosInstance.get(endpoint, {
        params: {
            nresults: maxTextResults, // Replace 'q' with the actual query parameter name the API expects
            // apiKey: 'API_KEY' // Include API key if necessary
        },
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },

    });
    ret = await response.data
    console.log(ret)
    return ret;
}

// calls Search api to get images
async function fetchSearchserviceFinesearch(positiveIndices,negativeIndices,dataset,searcher,numNegSamples,negativeWeight) {
    let url = `%ENDPOINT_PROTOCOL%://%ENDPOINT_SEARCHSERVICE%/search/${dataset}/${searcher}/?n_nonrare_samples=${numNegSamples}&nresults=${maxImages}&negative_weight=${negativeWeight}`
    try {
        const response = await axiosInstance.post(url, {
            idxs_rare: positiveIndices, 
            idxs_nonrare: negativeIndices
        }, {
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        });

        return response.data;
    } catch (error) {
        console.error("Error in fetchSearchserviceFinesearch:", error);
        // Handle error appropriately
        throw error;
    }
}




// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

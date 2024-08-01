const express = require('express');
const app = express();
const path = require('path');
const axios = require('axios');
const bodyParser = require('body-parser');
require('dotenv').config();

// Parse datasets from environment variable
const parseDatasets = (datasets) => {
    return datasets.split(',').reduce((acc, dataset) => {
        const [name, short] = dataset.split(':');
        acc[name] = short;
        return acc;
    }, {});
};

// dOptions from environment variable
const dOptions = parseDatasets(process.env.DATASETS);

const sOptions = {"DBranches":"db","DBranches Ensemble":"ens","Decision Tree":"dtree","Random Forest":"rf"};

const data_size = 150;
const maxTextResults = 60;
const maxNegSamples = process.env.DATASET_MAX_NUM_NEG;
const negSamples = process.env.DATASET_NUM_NEG;
const maxImages = 1000;
const endpoint_data = `${process.env.ENDPOINT_PROTOCOL}://${process.env.ENDPOINT_DATA}/image/`;
const demo = process.env.DEMO;
const negativeWeight = 1;

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
    const datasetOptions = Object.keys(dOptions);
    const defaultDataset = datasetOptions[0]; // Default dataset name

    res.render('index', { datasetOptions: datasetOptions, defaultDataset: defaultDataset, demo: demo });
});

app.get('/search', async (req, res) => {
    const searchOptions = Object.keys(sOptions);
    const defaultSearcher = searchOptions[0];
    const searchQuery = req.query.query;
    const dataset_key = req.query.dataset;
    const dataset = dOptions[dataset_key];

    try {
        const results = await fetchSearchserviceTextSearch(searchQuery, dataset, maxTextResults);
        const imageIndexes = results.results;
        const time = results.time.toFixed(3);

        const images = imageIndexes.map(idx =>
            [idx, `${endpoint_data}${dataset}/${idx}?size=${data_size}`]
        );

        console.log(images);

        res.render('results', {
            images: images,
            searchDataset: dataset,
            queryTime: time,
            searchOptions: searchOptions,
            defaultSearcher: defaultSearcher,
            maxNegSamples: maxNegSamples,
            negSamples: negSamples,
            demo: demo,
            negativeWeight: negativeWeight
        });
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

    console.log("Finetune:", positiveIndices);
    console.log("Finetune: ", negativeIndices);

    try {
        const results = await fetchSearchserviceFinesearch(positiveIndices, negativeIndices, dataset, searcher, numNegSamples, negativeWeight);
        console.log(results);
        const imageIndexes = results.results;
        const time = results.time.toFixed(3);
        const nresults = results.nresults;
        const sql_statements = results.sql_statements;

        const images = imageIndexes.map(idx =>
            [idx, `${endpoint_data}${dataset}/${idx}?size=${data_size}`]
        );

        res.json({ images: images, queryTime: time, nresults: nresults, sql_statements: sql_statements });
    } catch (error) {
        console.error('API error:', error);
        res.render('error', { message: 'Error fetching images' });
    }
});

async function fetchSearchserviceTextSearch(searchText, dataset, maxTextResults) {
    const endpoint = `${process.env.ENDPOINT_PROTOCOL}://${process.env.ENDPOINT_SEARCHSERVICE}/search_text/${dataset}/${searchText}`;
    console.log(endpoint);
    const response = await axiosInstance.get(endpoint, {
        params: {
            nresults: maxTextResults
        },
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
    });
    return response.data;
}

async function fetchSearchserviceFinesearch(positiveIndices, negativeIndices, dataset, searcher, numNegSamples, negativeWeight) {
    const url = `${process.env.ENDPOINT_PROTOCOL}://${process.env.ENDPOINT_SEARCHSERVICE}/search/${dataset}/${searcher}/?n_nonrare_samples=${numNegSamples}&nresults=${maxImages}&negative_weight=${negativeWeight}`;
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
        throw error;
    }
}

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

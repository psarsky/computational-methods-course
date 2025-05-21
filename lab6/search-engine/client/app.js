document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const useSvdCheckbox = document.getElementById('use-svd');
    const kSvdInput = document.getElementById('k-svd');
    const kResultsInput = document.getElementById('k-results');
    const saveConfigButton = document.getElementById('save-config');
    const statusElement = document.getElementById('status');
    const searchTimeElement = document.getElementById('search-time');
    const resultsContainer = document.getElementById('results-container');
    const documentView = document.getElementById('document-view');
    const documentContent = document.getElementById('document-content');
    const backButton = document.getElementById('back-button');

    const API_BASE_URL = '/api';

    checkEngineStatus();

    searchButton.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    saveConfigButton.addEventListener('click', updateConfiguration);
    backButton.addEventListener('click', () => {
        documentView.classList.add('hidden');
        resultsContainer.style.display = 'block';
    });

    /**
     * Checks the current status of the search engine.
     * @returns {void}
     */
    function checkEngineStatus() {
        statusElement.textContent = 'Checking search engine status...';
        statusElement.className = 'status';
        fetch(`${API_BASE_URL}/status`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'ready') {
                    statusElement.textContent = 'Search engine is ready.';
                    statusElement.classList.add('ready');
                } else {
                    statusElement.textContent = 'Search engine is initializing. Please wait...';
                    statusElement.classList.add('initializing');
                    setTimeout(checkEngineStatus, 3000);
                }                useSvdCheckbox.checked = data.use_svd;
                const kSvdValue = data.k_svd.toString();
                for (let i = 0; i < kSvdInput.options.length; i++) {
                    if (kSvdInput.options[i].value === kSvdValue) {
                        kSvdInput.selectedIndex = i;
                        break;
                    }
                }
                
                kResultsInput.value = data.k_results;
            })
            .catch(error => {
                statusElement.textContent = `Error: ${error.message}`;
                statusElement.className = 'status error';
            });
    }

    /**
     * Updates the search engine configuration with UI values.
     * @function
     * @returns {void}
     */
    function updateConfiguration() {
        const config = {
            use_svd: useSvdCheckbox.checked,
            k_svd: parseInt(kSvdInput.value),
            k_results: parseInt(kResultsInput.value)
        };
        fetch(`${API_BASE_URL}/config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'config_updated') {
                    statusElement.textContent = 'Configuration updated successfully.';
                    statusElement.className = 'status ready';
                    setTimeout(checkEngineStatus, 1000);
                } else if (data.error) {
                    statusElement.textContent = `Error: ${data.error}`;
                    statusElement.className = 'status error';
                }
            })
            .catch(error => {
                statusElement.textContent = `Error: ${error.message}`;
                statusElement.className = 'status error';
            });
    }
    /**
     * Performs a search based on the current query input value.
     * @returns {void}
     */
    function performSearch() {
        const query = searchInput.value.trim();
        if (!query) {
            statusElement.textContent = 'Please enter a search query.';
            statusElement.className = 'status error';
            return;
        }
        resultsContainer.innerHTML = '<p>Searching...</p>';
        searchTimeElement.textContent = '';
        fetch(`${API_BASE_URL}/search?q=${encodeURIComponent(query)}`)
            .then(async response => {
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.error || 'Search failed');
                }
                return response.json();
            })
            .then(data => {
                searchTimeElement.textContent = `Found ${data.results_count} results in ${data.time.toFixed(3)} seconds`;
                displaySearchResults(data.results);
            })
            .catch(error => {
                resultsContainer.innerHTML = '';
                statusElement.textContent = `Error: ${error.message}`;
                statusElement.className = 'status error';
                if (error.message.includes('initializing')) {
                    setTimeout(checkEngineStatus, 2000);
                }
            });
    }
    /**
     * Displays search results in the results container.
     * @param {Array<Object>} results - The search results to display
     * @param {number} results[].doc_id - Document ID
     * @param {string} results[].title - Document title
     * @param {string} results[].url - Document URL
     * @param {number} results[].similarity - Similarity score (0-1)
     * @returns {void}
     */
    function displaySearchResults(results) {
        resultsContainer.innerHTML = '';
        if (results.length === 0) {
            resultsContainer.innerHTML = '<p>No results found.</p>';
            return;
        }
        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            const title = document.createElement('div');
            title.className = 'result-title';
            title.textContent = result.title;
            const url = document.createElement('div');
            url.className = 'result-url';
            const link = document.createElement('a');
            link.href = result.url;
            link.textContent = result.url;
            link.target = '_blank';
            url.appendChild(link);
            const similarity = document.createElement('div');
            similarity.className = 'result-similarity';
            similarity.textContent = `Similarity: ${result.similarity.toFixed(4)}`;
            const viewButton = document.createElement('button');
            viewButton.className = 'result-view-btn';
            viewButton.textContent = 'View Document';
            viewButton.addEventListener('click', () => {
                viewDocument(result.doc_id);
            });
            resultItem.appendChild(title);
            resultItem.appendChild(url);
            resultItem.appendChild(similarity);
            resultItem.appendChild(viewButton);
            resultsContainer.appendChild(resultItem);
        });
    }
    /**
     * Fetches and displays a document's full content.
     * @param {number} docId - The ID of the document to view
     * @returns {void}
     */
    function viewDocument(docId) {
        fetch(`${API_BASE_URL}/document/${docId}`)
            .then(async response => {
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.error || 'Failed to load document');
                }
                return response.json();
            })
            .then(doc => {
                resultsContainer.style.display = 'none';
                documentView.classList.remove('hidden');
                documentContent.innerHTML = '';
                const title = document.createElement('h2');
                title.className = 'document-title';
                title.textContent = doc.title;
                const url = document.createElement('div');
                url.className = 'document-url';
                const link = document.createElement('a');
                link.href = doc.url;
                link.textContent = doc.url;
                link.target = '_blank';
                url.appendChild(link);
                const content = document.createElement('div');
                content.className = 'document-content';
                content.textContent = doc.content;
                documentContent.appendChild(title);
                documentContent.appendChild(url);
                documentContent.appendChild(content);
            })
            .catch(error => {
                statusElement.textContent = `Error: ${error.message}`;
                statusElement.className = 'status error';
            });
    }
});

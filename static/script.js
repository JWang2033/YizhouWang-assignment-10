document.addEventListener("DOMContentLoaded", () => {
  // Get references to the input field and button
  const searchButton = document.getElementById("search-button");
  const queryInput = document.getElementById("query-input");

  // Image search elements
  const imageSearchButton = document.getElementById("image-search-button");
  const imageInput = document.getElementById("image-input");

  // Hybrid
  const hybridSearchButton = document.getElementById("hybrid-search-button");
  const hybridQueryInput = document.getElementById("hybrid-query-input");
  const hybridImageInput = document.getElementById("hybrid-image-input");
  const lambdaInput = document.getElementById("lambda-input");
  const lambdaValue = document.getElementById("lambda-value");
  // Update lambda value dynamically
  if (lambdaInput && lambdaValue) {
    lambdaInput.addEventListener("input", () => {
      lambdaValue.textContent = lambdaInput.value;
    });
  }


  // text search
  // Check if the elements exist
  if (searchButton && queryInput) {
    searchButton.addEventListener("click", () => {
      const query = queryInput.value.trim(); // Get the query text
      if (query) {
        performTextSearch(query); // Call the function to perform the text search
      } else {
        alert("Please enter a valid query."); // Alert if the query is empty
      }
    });
  } else {
    console.error("Search button or query input not found in DOM.");
  }

  // image search
  if (imageSearchButton && imageInput) {
    imageSearchButton.addEventListener("click", () => {
        const imageFile = imageInput.files[0]; // Get the selected file
        if (imageFile) {
            performImageSearch(imageFile); // Perform image search
        } else {
            alert("Please upload an image.");
        }
    });
  } else {
      console.error("Image search button or image input not found in DOM.");
  }

  // Hybrid Search
  if (hybridSearchButton && hybridQueryInput && hybridImageInput && lambdaInput) {
    hybridSearchButton.addEventListener("click", () => {
      const textQuery = hybridQueryInput.value.trim();
      const imageFile = hybridImageInput.files[0];
      const lambda = parseFloat(lambdaInput.value);

      if (textQuery && imageFile && lambda >= 0 && lambda <= 1) {
        performHybridSearch(textQuery, imageFile, lambda);
      } else {
        alert("Please provide a valid text query, image, and lambda value.");
      }
    });
  }

});



/**
 * Perform a text-to-image search using the API.
 * @param {string} query - The text query for the search.
 */
async function performTextSearch(query) {
  try {
    // Send a POST request to the /search/text endpoint
    const response = await fetch("/search/text", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    // Handle HTTP errors
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Parse the JSON response
    const data = await response.json();
    console.log("Response data:", data);

    // Display result in the result container
    const resultContainer = document.getElementById("result-container");
    if (resultContainer) {
      resultContainer.innerHTML = `
        <p><strong>Retrieved Image:</strong> ${data.retrieved_image}</p>
        <p><strong>Similarity Score:</strong> ${data.similarity_score.toFixed(4)}</p>
        <img src="${data.retrieved_image}" alt="Retrieved Image" style="max-width: 100%; border: 1px solid #ccc; padding: 10px;">
      `;
    }
  } catch (error) {
    console.error("Error during text search:", error);
    alert("An error occurred while performing the search. Please try again.");
  }
}

/**
 * Perform an image-to-image search using the API.
 * @param {File} imageFile - The uploaded image file.
 */
async function performImageSearch(imageFile) {
  try {
    const formData = new FormData();
    formData.append("image", imageFile);

    const response = await fetch("/search/image", {
      method: "POST",
      body: formData,
    });

    // Handle HTTP errors
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Parse the JSON response
    const data = await response.json();
    console.log("Response data:", data);

    // Display result in the result container
    const resultContainer = document.getElementById("result-container");
    if (resultContainer) {
      resultContainer.innerHTML = `
        <p><strong>Retrieved Image:</strong> ${data.retrieved_image}</p>
        <p><strong>Similarity Score:</strong> ${data.similarity_score.toFixed(4)}</p>
        <img src="/${data.output_image_path}" alt="Retrieved Image" style="max-width: 100%; border: 1px solid #ccc; padding: 10px;">
      `;
    }
  } catch (error) {
    console.error("Error during image search:", error);
    alert("An error occurred during the image search. Please try again.");
  }
}

/**
 * Perform a hybrid search using the API.
 * @param {string} textQuery - The text query for the search.
 * @param {File} imageFile - The uploaded image file.
 * @param {number} lambda - The lambda value for weighting.
 */
async function performHybridSearch(textQuery, imageFile, lambda) {
  try {
    const formData = new FormData();
    formData.append("query", textQuery);
    formData.append("image", imageFile);
    formData.append("lambda", lambda);

    const response = await fetch("/search/hybrid", {
      method: "POST",
      body: formData,
    });

    // Handle HTTP errors
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Parse the JSON response
    const data = await response.json();
    console.log("Response data:", data);

    // Display result in the result container
    const resultContainer = document.getElementById("result-container");
    if (resultContainer) {
      resultContainer.innerHTML = `
        <p><strong>Retrieved Image:</strong> ${data.retrieved_image}</p>
        <p><strong>Similarity Score:</strong> ${data.similarity_score.toFixed(4)}</p>
        <img src="/${data.output_image_path}" alt="Retrieved Image" style="max-width: 100%; border: 1px solid #ccc; padding: 10px;">
      `;
    }
  } catch (error) {
    console.error("Error during hybrid search:", error);
    alert("An error occurred during the hybrid search. Please try again.");
  }
}

/**
 * Display the search results in the result container.
 * @param {Object} data - The response data from the server.
 */
function displayResult(data) {
  const resultContainer = document.getElementById("result-container");
  if (resultContainer) {
    resultContainer.innerHTML = `
      <p><strong>Retrieved Image:</strong> ${data.retrieved_image}</p>
      <p><strong>Similarity Score:</strong> ${data.similarity_score.toFixed(4)}</p>
      <img src="/${data.output_image_path}" alt="Retrieved Image" style="max-width: 100%; border: 1px solid #ccc; padding: 10px;">
    `;
  }
}

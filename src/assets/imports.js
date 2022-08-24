function docReady(fn) {
  // see if DOM is already available
  if (
    document.readyState === "complete" ||
    document.readyState === "interactive"
  ) {
    // call on next available tick
    setTimeout(fn, 1);
  } else {
    document.addEventListener("DOMContentLoaded", fn);
  }
}

// adds the apps HTML pages to the DOM
document.addEventListener("DOMContentLoaded", async function () {
  const links = document.querySelectorAll('link[rel="import"]');

  let contentIndex = document.querySelector("#content");

  // Import and add each page to the DOM
  for (const link of links) {
    let doc = await fetch(link.href, {
      headers: {
        "Content-Type": "text/html",
      },
    });

    let content = await doc.text();

    //Add the HTML Section to the #content container
    contentIndex.innerHTML += content;
  }

  //sleep for a second to allow templates to be inserted into the DOM
  await new Promise((r) => setTimeout(r, 1000));

  //Synchronously include js files
  includeJavaScriptFile("./assets/ex-links.js");
  includeJavaScriptFile("./assets/nav.js");
  includeJavaScriptFile("./assets/demo-btns.js");
  includeJavaScriptFile("./preload.js");
  includeJavaScriptFile("./scripts/others/renderer.js");
  includeJavaScriptFile("./scripts/others/tab-effects.js");
  includeJavaScriptFile("./scripts/disseminate/disseminate.js");
  includeJavaScriptFile("./scripts/disseminate/prePublishingReview.js");
  includeJavaScriptFile("./scripts/manage-dataset/manage-dataset.js");
  includeJavaScriptFile("./scripts/metadata-files/datasetDescription.js");
  includeJavaScriptFile("./scripts/organize-dataset/curate-functions.js");
  includeJavaScriptFile("./scripts/organize-dataset/organizeDS.js");
  includeJavaScriptFile("./scripts/metadata-files/manifest.js");
  includeJavaScriptFile("./scripts/metadata-files/readme-changes.js");
  includeJavaScriptFile("./scripts/metadata-files/subjects-samples.js");
  includeJavaScriptFile("./scripts/metadata-files/submission.js");
  includeJavaScriptFile("./scripts/guided-mode/lottieJSON.js");
  includeJavaScriptFile("./scripts/guided-mode/guided-curate-dataset.js");
});

//Synchronously attaches a javascript file to the DOM
const includeJavaScriptFile = (jsFilePath) => {
  var js = document.createElement("script");
  js.type = "text/javascript";
  js.src = jsFilePath;
  document.body.appendChild(js);
};

const sleep = (milliseconds) => {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
};

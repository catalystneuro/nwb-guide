import { updateURLParams } from "../../utils/url.js";
import { guidedProgressFilePath } from "../globals.js";
import { fs } from "../../utils/electron.js";
import { joinPath } from "../globals";
import { get, hasEntry } from "./index.js";

export const rename = (newDatasetName, previousDatasetName) => {
    //If updating the dataset, update the old banner image path with a new one
    if (previousDatasetName) {
        if (previousDatasetName === newDatasetName) return;

        if (hasEntry(newDatasetName))
            throw new Error("An existing project already exists with that name. Please choose a different name.");

        // update old progress file with new dataset name
        const oldProgressFilePath = `${guidedProgressFilePath}/${previousDatasetName}.json`;
        const newProgressFilePath = `${guidedProgressFilePath}/${newDatasetName}.json`;
        fs.renameSync(oldProgressFilePath, newProgressFilePath);
        
    } else throw new Error("No previous project name provided");
};

export const updateAppProgress = (
    pageId,
    dataOrProjectName = {},
    projectName = typeof dataOrProjectName === "string" ? dataOrProjectName : undefined
) => {
    const transitionOffPipeline = pageId && pageId.slice(0, 2) !== "//";

    if (transitionOffPipeline) {
        updateURLParams({ project: undefined });
        return; // Only save last page if within the conversion workflow
    }

    if (projectName) updateURLParams({ project: projectName });

    // Is a project name
    if (dataOrProjectName === projectName) updateFile(dataOrProjectName, (data) => (data["page-before-exit"] = pageId));
    // Is a data object
    else dataOrProjectName["page-before-exit"] = pageId;
};

//Destination: HOMEDIR/NWB_GUIDE/pipelines
export const updateFile = (projectName, callback) => {
    let data = get(projectName);

    if (callback) {
        const result = callback(data);
        if (result && typeof result === "object") data = result;
    }

    data["last-modified"] = new Date(); // Always update the last modified time

    var guidedFilePath = joinPath(guidedProgressFilePath, projectName + ".json");

    console.log(guidedProgressFilePath)

    // Save the file through the available mechanisms
    if (!fs.existsSync(guidedProgressFilePath)) fs.mkdirSync(guidedProgressFilePath, { recursive: true }); //create progress folder if one does not exist
    fs.writeFileSync(guidedFilePath, JSON.stringify(data, null, 2));
};

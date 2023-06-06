import { LitElement, html } from "lit";
import { openProgressSwal, runConversion } from "./guided-mode/options/utils.js";
import { get, save } from "../../progress.js";
import { dismissNotification, notify } from "../../dependencies/globals.js";
import { merge, randomizeElements, mapSessions } from "./utils.js";

import { ProgressBar } from "../ProgressBar";

export class Page extends LitElement {
    // static get styles() {
    //     return useGlobalStyles(
    //         componentCSS,
    //         (sheet) => sheet.href && sheet.href.includes("bootstrap"),
    //         this.shadowRoot
    //     );
    // }

    info = { globalState: {} };

    constructor(info = {}) {
        super();
        Object.assign(this.info, info);

        this.style.height = "100%";
        this.style.color = "black";
    }

    createRenderRoot() {
        return this;
    }

    query = (input) => {
        return (this.shadowRoot ?? this).querySelector(input);
    };

    onSet = () => {}; // User-defined function

    set = (info) => {
        if (info) {
            Object.assign(this.info, info);
            this.onSet();
            this.requestUpdate();
        }
    };

    #notifications = [];

    dismiss = (notification) => {
        if (notification) dismissNotification(notification);
        else {
            this.#notifications.forEach((notification) => dismissNotification(notification));
            this.#notifications = [];
        }
    };

    notify = (...args) => {
        const note = notify(...args);
        this.#notifications.push(note);
    };

    onTransition = () => {}; // User-defined function
    updatePages = () => {}; // User-defined function

    save = (overrides) => save(this, overrides);

    load = (datasetNameToResume = new URLSearchParams(window.location.search).get("project")) =>
        (this.info.globalState = get(datasetNameToResume));

    merge = merge;

    addSession({ subject, session, info }) {
        if (!this.info.globalState.results[subject]) this.info.globalState.results[subject] = {};
        if (this.info.globalState.results[subject][session])
            throw new Error(`Session ${subject}/${session} already exists.`);
        info = this.info.globalState.results[subject][session] = info ?? {};
        if (!info.metadata) info.metadata = {};
        if (!info.source_data) info.source_data = {};
        return info;
    }

    removeSession({ subject, session }) {
        delete this.info.globalState.results[subject][session];
    }

    mapSessions = (callback) => mapSessions(callback, this.info.globalState);

    async runConversions(conversionOptions = {}, toRun, options = {}) {
        let original = toRun;
        if (!Array.isArray(toRun)) toRun = this.mapSessions();

        // Filter the sessions to run
        if (typeof original === "number") toRun = randomizeElements(toRun, original); // Grab a random set of sessions
        else if (typeof original === "string") toRun = toRun.filter(({ subject }) => subject === original);
        else if (typeof original === "function") toRun = toRun.filter(original);

        let results = [];

        const popup = await openProgressSwal({ title: `Running conversion`, ...options });

        const isMultiple = toRun.length > 1;

        let elements = {};
        // if (isMultiple) {
        popup.hideLoading();
        const element = popup.getHtmlContainer();
        element.innerText = "";

        const progressBar = new ProgressBar();
        elements.progress = progressBar;
        element.append(progressBar);
        // }

        let completed = 0;
        elements.progress.value = { b: completed, tsize: toRun.length };

        for (let info of toRun) {
            const { subject, session } = info;
            const file = `sub-${subject}/sub-${subject}_ses-${session}.nwb`;
            const relativePath = `${this.info.globalState.project.name}/${file}`;

            console.log("relativePath", relativePath);
            const result = await runConversion(
                {
                    nwbfile_path: relativePath,
                    overwrite: true, // We assume override is true because the native NWB file dialog will not allow the user to select an existing file (unless they approve the overwrite)
                    ...this.info.globalState.results[subject][session], // source_data and metadata are passed in here
                    ...conversionOptions, // Any additional conversion options override the defaults

                    interfaces: this.info.globalState.interfaces,
                },
                { swal: popup, ...options }
            ).catch((e) => {
                this.notify(e.message, "error");
                popup.close();
                throw e.message;
            });

            completed++;
            if (isMultiple) {
                const progressInfo = { b: completed, bsize: 1, tsize: toRun.length };
                elements.progress.value = progressInfo;
            }

            results.push(result);
        }

        popup.close();

        return results;
    }

    //   NOTE: Until the shadow DOM is supported in Storybook, we can't use this render function how we'd intend to.
    addPage = (id, subpage) => {
        if (!this.info.pages) this.info.pages = {};
        this.info.pages[id] = subpage;
        this.updatePages();
    };

    render() {
        return html`<slot></slot>`;
    }
}

customElements.get("nwbguide-page") || customElements.define("nwbguide-page", Page);

import { isStorybook } from "../../../../dependencies/globals";
import { JSONSchemaForm } from "../../../JSONSchemaForm.js";
import { InstanceManager } from "../../../InstanceManager.js";
import { ManagedPage } from "./ManagedPage.js";
import { onThrow } from "../../../../errors";
import { merge } from "../../utils";
import preprocessSourceDataSchema from "../../../../../../../schemas/source-data.schema";

import { createGlobalFormModal } from "../../../forms/GlobalFormModal";
import { header } from "../../../forms/utils";
import { Button } from "../../../Button.js";

import globalIcon from "../../../assets/global.svg?raw";

import { run } from "../options/utils.js";
import { getInfoFromId } from "./utils";
import { Modal } from "../../../Modal";

const propsToIgnore = {
    "*": {
        verbose: true,
        es_key: true,
        exclude_shanks: true,
        load_sync_channel: true,
        stream_id: true, // NOTE: May be desired for other interfaces
        nsx_override: true,
        combined: true,
        plane_no: true,
    },
};

export class GuidedSourceDataPage extends ManagedPage {
    constructor(...args) {
        super(...args);
        this.style.height = "100%"; // Fix main section
    }

    beforeSave = () => {
        merge(this.localState, this.info.globalState);
    };

    #globalButton = new Button({
        icon: globalIcon,
        label: "Edit Default Values",
        onClick: () => {
            this.#globalModal.form.results = structuredClone(this.info.globalState.project.SourceData ?? {});
            this.#globalModal.open = true;
        },
    });

    header = {
        controls: [this.#globalButton],
        subtitle:
            "Specify the file and folder locations on your local system for each interface, as well as any additional details that might be required.",
    };

    workflow = {
        multiple_sessions: {
            elements: [this.#globalButton],
        },
    };

    footer = {
        onNext: async () => {
            await this.save(); // Save in case the conversion fails

            for (let { form } of this.forms) await form.validate(); // Will throw an error in the callback

            // const previousResults = this.info.globalState.metadata.results

            await Promise.all(
                Object.values(this.forms).map(async ({ subject, session, form }) => {
                    const info = this.info.globalState.results[subject][session];

                    // NOTE: This clears all user-defined results
                    const result = await run(
                        `neuroconv/metadata`,
                        {
                            source_data: form.resolved, // Use resolved values, including global source data
                            interfaces: this.info.globalState.interfaces,
                        },
                        {
                            title: "Getting metadata for source data",
                            verbose: true,
                        }
                    ).catch((e) => {
                        this.notify(e.message, "error");
                        throw e;
                    });

                    if (isStorybook) return;

                    const { results: metadata, schema } = result;

                    // Merge arrays from generated pipeline data
                    if (info.metadata.__generated) {
                        const generated = info.metadata.__generated;
                        info.metadata = merge(merge(generated, metadata, { arrays: true }), info.metadata);
                    }

                    // Merge new results with old metadata
                    else merge(metadata, info.metadata);

                    // Mirror structure with metadata schema
                    const schemaGlobal = this.info.globalState.schema;
                    if (!schemaGlobal.metadata) schemaGlobal.metadata = {};
                    if (!schemaGlobal.metadata[subject]) schemaGlobal.metadata[subject] = {};
                    schemaGlobal.metadata[subject][session] = schema;
                })
            );

            await this.save(undefined, false); // Just save new raw values

            return this.to(1);
        },
    };

    createForm = ({ subject, session, info }) => {
        const hasMultipleSessions = this.workflow.multiple_sessions.value;

        const instanceId = `sub-${subject}/ses-${session}`;

        const schema = structuredClone(this.info.globalState.schema.source_data);
        delete schema.description;

        const form = new JSONSchemaForm({
            identifier: instanceId,
            schema: preprocessSourceDataSchema(schema),
            results: info.source_data,
            emptyMessage: "No source data required for this session.",
            ignore: propsToIgnore,
            globals: hasMultipleSessions ? this.info.globalState.project.SourceData : undefined,
            onOverride: (name) => {
                this.notify(`<b>${header(name)}</b> has been overridden with a global value.`, "warning", 3000);
            },
            // onlyRequired: true,
            onUpdate: () => (this.unsavedUpdates = "conversions"),
            onStatusChange: (state) => this.manager.updateState(instanceId, state),
            onThrow,
        });

        form.style.height = "100%";

        return {
            subject,
            session,
            form,
        };
    };

    #globalModal = null;

    connectedCallback() {
        super.connectedCallback();

        const schema = structuredClone(this.info.globalState.schema.source_data);
        delete schema.description;

        const modal = (this.#globalModal = createGlobalFormModal.call(this, {
            header: "Global Source Data",
            propsToRemove: {
                "*": {
                    ...propsToIgnore["*"],
                    folder_path: true,
                    file_path: true,
                    // NOTE: Still keeping plural path specifications for now
                },
            },
            key: "SourceData",
            schema,
            hasInstances: true,
        }));
        document.body.append(modal);
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        if (this.#globalModal) this.#globalModal.remove();
    }

    updated() {
        const dashboard = document.querySelector("nwb-dashboard");
        const page = dashboard.page;
    }

    render() {
        this.localState = { results: structuredClone(this.info.globalState.results ?? {}) };

        this.forms = this.mapSessions(this.createForm, this.localState.results);

        let instances = {};
        this.forms.forEach(({ subject, session, form }) => {
            if (!instances[`sub-${subject}`]) instances[`sub-${subject}`] = {};
            instances[`sub-${subject}`][`ses-${session}`] = form;
        });

        this.manager = new InstanceManager({
            header: "Sessions",
            // instanceType: 'Session',
            instances,
            controls: [
                {
                    name: "Check Alignment",
                    primary: true,
                    onClick: async (id) => {
                        const { globalState } = this.info;

                        const { subject, session } = getInfoFromId(id);

                        const souceCopy = structuredClone(globalState.results[subject][session].source_data);

                        const sessionInfo = {
                            interfaces: globalState.interfaces,
                            source_data: merge(globalState.project.SourceData, souceCopy),
                        };

                        const results = await run("neuroconv/alignment", sessionInfo, {
                            title: "Checking Alignment",
                            message: "Please wait...",
                        });

                        const header = document.createElement("div");
                        const h2 = document.createElement("h2");
                        Object.assign(h2.style, {
                            marginBottom: "10px",
                        });
                        h2.innerText = `Alignment Preview: ${subject}/${session}`;
                        const warning = document.createElement("small");
                        warning.innerHTML =
                            "<b>Warning:</b> This is just a preview. We do not currently have the features implemented to change the alignment of your interfaces.";
                        header.append(h2, warning);

                        const modal = new Modal({
                            header,
                        });

                        document.body.append(modal);

                        const content = document.createElement("div");
                        Object.assign(content.style, {
                            display: "flex",
                            flexDirection: "column",
                            gap: "20px",
                            padding: "20px",
                        });

                        modal.append(content);

                        const flatTimes = Object.values(results)
                            .map((interfaceTimestamps) => {
                                return [interfaceTimestamps[0], interfaceTimestamps.slice(-1)[0]];
                            })
                            .flat()
                            .filter((timestamp) => !isNaN(timestamp));

                        const minTime = Math.min(...flatTimes);
                        const maxTime = Math.max(...flatTimes);

                        const normalizeTime = (time) => (time - minTime) / (maxTime - minTime);
                        const normalizeTimePct = (time) => `${normalizeTime(time) * 100}%`;

                        for (let name in results) {
                            const container = document.createElement("div");
                            const label = document.createElement("label");
                            label.innerText = name;
                            container.append(label);

                            const data = results[name];

                            const barContainer = document.createElement("div");
                            Object.assign(barContainer.style, {
                                height: "10px",
                                width: "100%",
                                marginTop: "5px",
                                border: "1px solid lightgray",
                                position: "relative",
                            });

                            if (data.length) {
                                const firstTime = data[0];
                                const lastTime = data[data.length - 1];

                                label.innerText += ` (${firstTime.toFixed(2)} - ${lastTime.toFixed(2)} sec)`;

                                const firstTimePct = normalizeTimePct(firstTime);
                                const lastTimePct = normalizeTimePct(lastTime);

                                const width = `calc(${lastTimePct} - ${firstTimePct})`;

                                const bar = document.createElement("div");

                                Object.assign(bar.style, {
                                    position: "absolute",

                                    left: firstTimePct,
                                    width: width,
                                    height: "100%",
                                    background: "blue",
                                });

                                barContainer.append(bar);
                            } else {
                                barContainer.style.background =
                                    "repeating-linear-gradient(45deg, lightgray, lightgray 10px, white 10px, white 20px)";
                            }

                            container.append(barContainer);

                            content.append(container);
                        }

                        modal.open = true;
                    },
                },
            ],
            // onAdded: (path) => {

            //   let details = this.getDetails(path)

            //   const info = this.addSession(details)

            //   const form = this.createForm({
            //     ...details,
            //     info
            //   })

            //   this.forms.push(form)

            //   return {
            //     key: `sub-${details.subject}/ses-${details.session}`,
            //     value: form.form
            //   }
            // },
            // onRemoved: (_, path) => {
            //   let details = this.getDetails(path)
            //   this.removeSession(details)
            // }
        });

        return this.manager;
    }
}

customElements.get("nwbguide-guided-sourcedata-page") ||
    customElements.define("nwbguide-guided-sourcedata-page", GuidedSourceDataPage);

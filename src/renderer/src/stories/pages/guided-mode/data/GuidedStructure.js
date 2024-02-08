import { html } from "lit";
import { Page } from "../../Page.js";

// For Multi-Select Form
import { Button } from "../../../Button.js";
import { supportedInterfaces } from "../../../../globals.js";
import { Search } from "../../../Search.js";
import { Modal } from "../../../Modal";
import { List } from "../../../List";
import { baseUrl } from "../../../../server/globals";

const defaultEmptyMessage = "No formats selected";

const categories = [
    {
        test: /.*Interface.*/,
        value: "Single-Stream Interfaces",
    },
    {
        test: /.*Converter.*/,
        value: "Multi-Stream Converters",
    },
];

export class GuidedStructurePage extends Page {
    constructor(...args) {
        super(...args);

        // Handle Search Bar Interactions
        this.search.list.style.position = "unset";
        this.search.onSelect = (item) => {
            this.list.add(item);
            this.searchModal.toggle(false);
        };

        this.addButton.innerText = "Add Format";
        this.addButton.onClick = () => {
            this.searchModal.toggle(true);
        };

        this.searchModal.appendChild(this.search);
    }

    header = {
        subtitle: "List all the data formats in your dataset.",
    };

    search = new Search({
        disabledLabel: "Not supported",
        headerStyles: {
            padding: "15px",
        },
    });

    list = new List({
        emptyMessage: defaultEmptyMessage,
        onChange: () => (this.unsavedUpdates = "conversions"),
    });

    addButton = new Button();

    searchModal = new Modal({
        width: "100%",
        height: "100%",
    });

    getSchema = async () => {
        const interfaces = { ...this.list.object };

        const schema =
            Object.keys(interfaces).length === 0
                ? {}
                : await fetch(`${baseUrl}/neuroconv/schema`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify(interfaces),
                  }).then((res) => res.json());

        let schemas = this.info.globalState.schema;
        if (!schemas) schemas = this.info.globalState.schema = {};

        schemas.source_data = schema;
        this.unsavedUpdates = true;
    };

    beforeSave = async () => {
        this.info.globalState.interfaces = { ...this.list.object };

        // Remove extra interfaces from results
        if (this.info.globalState.results) {
            this.mapSessions(({ info }) => {
                Object.keys(info.source_data).forEach((key) => {
                    if (!this.info.globalState.interfaces[key]) delete info.source_data[key];
                });
            });
        }

        await this.save(undefined, false); // Interrim save, in case the schema request fails
        await this.getSchema();
    };

    footer = {
        onNext: async () => {
            if (!this.info.globalState.schema) await this.getSchema(); // Initialize schema
            this.to(1);
        },
    };

    async updated() {
        const { interfaces = {} } = this.info.globalState;

        this.list.emptyMessage = "Loading valid formats...";

        this.search.options = await fetch(`${baseUrl}/neuroconv`)
            .then((res) => res.json())
            .then((json) =>
                Object.entries(json).map(([key, value]) => {
                    const category = categories.find(({ test }) => test.test(key))?.value;

                    let displayName = key;

                    // replace the keys in the schema with a new string that is the original key with the string "New" appended at the end
                    // displayName = displayName.replace("Recording", " Recording");
                    // displayName = displayName.replace("Sorting", " Sorting");
                    // displayName = displayName.replace("Imaging", " Imaging");
                    // displayName = displayName.replace("Segmentation", " Segmentation");
                    // displayName = displayName.replace("Data", " Data");
                    // displayName = displayName.replace("LFP", " LFP");
                    // displayName = displayName.replace("NIDQ", " NIDQ");
                    // displayName = displayName.replace("Tiff", " Tiff");
                    // displayName = displayName.replace("SinglePlane", " Single Plane");
                    // displayName = displayName.replace("MultiPlane", " Multi Plane");
                    // if (displayName.endsWith("Interface")) displayName = displayName.replace("Interface", "");
                    // else if (displayName.endsWith("Converter")) displayName = displayName.replace("Converter", " (All Data)");
                    // else if (displayName.endsWith("ConverterPipe")) displayName = displayName.replace("ConverterPipe", " (All Data)");

                    displayName = displayName.trim();

                    const interfaceName = value.name;

                    return {
                        ...value,
                        key: displayName,
                        value: interfaceName,
                        category,
                        // displayName: displayName,
                        disabled: !supportedInterfaces.includes(interfaceName),
                    }; // Has label and keywords property already
                })
            )
            .catch((error) => console.error(error));

        this.list.emptyMessage = defaultEmptyMessage;

        for (const [key, name] of Object.entries(interfaces)) {
            let found = this.search.options?.find((item) => item.value === name);

            // If not found, spoof based on the key and names provided previously
            if (!found) {
                found = {
                    key,
                    label: name.replace("Interface", ""),
                    value: name,
                };
            }

            this.list.add({ ...found, key }); // Add previously selected items
        }

        this.addButton.removeAttribute("hidden");
        super.updated(); // Call if updating data
    }

    render() {
        // Reset list
        this.list.style.display = "inline-block";
        this.list.clear();
        this.addButton.setAttribute("hidden", "");

        return html`
            <div style="width: 100%; display: flex; flex-direction: column; align-items: center;">
                ${this.list} ${this.addButton}
            </div>
            ${this.searchModal}
        `;
    }
}

customElements.get("nwbguide-guided-structure-page") ||
    customElements.define("nwbguide-guided-structure-page", GuidedStructurePage);

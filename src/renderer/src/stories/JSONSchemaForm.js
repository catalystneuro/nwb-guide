import { LitElement, css, html } from "lit";
import { unsafeHTML } from "lit/directives/unsafe-html.js";

import { Accordion } from "./Accordion";

import { checkStatus } from "../validation";
import { header, replaceRefsWithValue } from "./forms/utils";
import { resolve } from "../promises";
import { merge } from "./pages/utils";
import { resolveProperties } from "./pages/guided-mode/data/utils";

import { JSONSchemaInput, getEditableItems } from "./JSONSchemaInput";
import { InspectorListItem } from "./preview/inspector/InspectorList";

const encode = (str) => {
    try {
        document.querySelector(`#${str}`);
        return str;
    } catch {
        return btoa(str).replace(/\+|\/|\=/g, "_");
    }
};

const additionalPropPattern = "additional";

const templateNaNMessage = `<br/><small>Type <b>NaN</b> to represent an unknown value.</small>`;

import { Validator } from "jsonschema";
import { successHue, warningHue, errorHue } from "./globals";
import { Button } from "./Button";
import { Tabs } from "./tabs/Tabs";
import { TabItem } from "./tabs/TabItem";
import { ContextMenu } from "./table/ContextMenu";

var validator = new Validator();

const isObject = (item) => {
    return item && typeof item === "object" && !Array.isArray(item);
};

export const getIgnore = (o, path) => {
    if (typeof path === "string") path = path.split(".");
    return path.reduce((acc, key) => {
        const info = acc[key] ?? {};

        return {
            ...info,
            "*": { ...(acc["*"] ?? {}), ...(info["*"] ?? {}) }, // Accumulate ignore values
        };
    }, o);
};

const selfRequiredSymbol = Symbol();

const componentCSS = `

    * {
      box-sizing: border-box;
    }

    :host {
      display: inline-block;
      width: 100%;
      height: 100%;
    }

    #empty {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        width: 100%;
        color: gray;
    }


    p {
      margin: 0 0 1em;
      line-height: 1.4285em;
    }

    *:first-child jsonschema-input {
      margin: 0;
    }

    .link {
      margin-top: 20px;
      border: 1px solid black;
      border-radius: 4px;
      position: relative;
    }

    .link > div {
      padding: 20px;
    }

    .link::before {
      box-sizing: border-box;
      display: block;
      width: 100%;
      color: white;
      background: black;
      padding: 10px;
      content: ''attr(data-name)'';
      font-weight: bold;
    }

    .link::after {
        box-sizing: border-box;
        display: block;
        width: 10px;
        height: 10px;
        background: hsl(${successHue}, 100%, 70%) !important;
        border-radius: 50%;
        position: absolute;
        top: 0;
        right: 0;
        content: '';
        margin: 15px;
      }

    .link.error::after {
        background: hsl(${errorHue}, 100%, 70%) !important;
    }

    .link.warning::after {
        background: hsl(${warningHue}, 100%, 70%) !important;
    }

    hr {
    margin: 1em 0 1.5em 0;
    }

    pre {
        white-space: pre-wrap;       /* Since CSS 2.1 */
        white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
        white-space: -pre-wrap;      /* Opera 4-6 */
        white-space: -o-pre-wrap;    /* Opera 7 */
        word-wrap: break-word;       /* Internet Explorer 5.5+ */
        font-family: unset;
        color: DimGray;
    }

    h4 {
        margin: 0;
        margin-bottom: 5px;
        padding-bottom: 5px;
        border-bottom: 1px solid gainsboro;
    }

    .guided--text-input-instructions {
        font-size: 13px;
        width: 100%;
        padding-top: 4px;
        color: dimgray !important;
        margin: 0 0 1em;
        line-height: 1.4285em;
    }

    nwb-accordion {
        margin-bottom: 0.5em;
    }

    [disabled]{
        opacity: 0.5;
        pointer-events: none;
    }

    small {
        font-size: 0.8em;
    }

`;

document.addEventListener("dragover", (dragEvent) => {
    dragEvent.preventDefault();
    dragEvent.stopPropagation();
});

export class JSONSchemaForm extends LitElement {
    static get styles() {
        return css([componentCSS]);
    }

    static get properties() {
        return {
            schema: { type: Object, reflect: false },
            results: { type: Object, reflect: false },
            required: { type: Object, reflect: false },
            dialogType: { type: String, reflect: false },
            dialogOptions: { type: Object, reflect: false },
            globals: { type: Object, reflect: false },
            validateEmptyValues: { type: Boolean, reflect: true },
        };
    }

    base = [];
    forms = {};
    inputs = [];

    tables = {};
    #nErrors = 0;
    #nWarnings = 0;

    getActiveForms = () =>
        Object.entries(this.forms)
            .filter(([k, v]) => !this.tabs[k] || !this.tabs[k].disabled)
            .map(([_, v]) => v);

    get nErrors() {
        const reducer = (acc, item) => acc + item.nErrors;
        const tableErrors = Object.values(this.tables).reduce(reducer, 0);
        const activeFormErrors = this.getActiveForms().reduce(reducer, 0);
        return this.#nErrors + tableErrors + activeFormErrors;
    }

    get nWarnings() {
        const reducer = (acc, item) => acc + item.nWarnings;
        const tableWarnings = Object.values(this.tables).reduce(reducer, 0);
        const activeFormWarnings = this.getActiveForms().reduce(reducer, 0);
        return this.#nWarnings + tableWarnings + activeFormWarnings;
    }

    #toggleRendered;
    #rendered;
    #updateRendered = (force) =>
        force || this.#rendered === true
            ? (this.#rendered = new Promise(
                  (resolve) => (this.#toggleRendered = () => resolve((this.#rendered = true)))
              ))
            : this.#rendered;

    resolved = {}; // Keep track of actual resolved values—not just what the user provides as results

    constructor(props = {}) {
        super();

        this.#rendered = this.#updateRendered(true);

        this.identifier = props.identifier;
        this.schema = props.schema ?? {};
        this.results = (props.base ? structuredClone(props.results) : props.results) ?? {}; // Deep clone results in nested forms
        this.globals = props.globals ?? {};

        this.ignore = props.ignore ?? {};
        this.required = props.required ?? {};
        this.dialogOptions = props.dialogOptions;
        this.dialogType = props.dialogType;
        this.deferLoading = props.deferLoading ?? false;

        this.controls = props.controls ?? {};

        this.transformErrors = props.transformErrors;

        this.emptyMessage = props.emptyMessage ?? "No properties to render";

        this.onlyRequired = props.onlyRequired ?? false;
        this.showLevelOverride = props.showLevelOverride ?? false;

        this.groups = props.groups ?? []; // NOTE: We assume properties only belong to one conditional requirement group

        this.validateEmptyValues = props.validateEmptyValues ?? true;

        if (props.onInvalid) this.onInvalid = props.onInvalid;
        if (props.sort) this.sort = props.sort;

        if (props.validateOnChange) this.validateOnChange = props.validateOnChange;
        if (props.onThrow) this.onThrow = props.onThrow;
        if (props.onLoaded) this.onLoaded = props.onLoaded;
        if (props.onUpdate) this.onUpdate = props.onUpdate;
        if (props.renderTable) this.renderTable = props.renderTable;
        if (props.renderCustomHTML) this.renderCustomHTML = props.renderCustomHTML;
        if (props.onOverride) this.onOverride = props.onOverride;

        if (props.onStatusChange) this.onStatusChange = props.onStatusChange;
        if (props.onStatusUpdate) this.onStatusUpdate = props.onStatusUpdate;

        if (props.base) this.base = props.base;
    }

    // Get the form element defined by the path (stops before table cells)
    getFormElement = (
        path,
        { forms, tables, inputs } = {
            forms: true,
            tables: true,
            inputs: true,
        }
    ) => {
        if (typeof path === "string") path = path.split(".");
        if (!path.length) return this;

        const name = path[0];
        const updatedPath = path.slice(1);

        const form = this.forms[name]; // Check forms
        if (!form) {
            const table = this.tables[name]; // Check tables
            if (table && tables) return table; // Skip table cells
        } else if (!updatedPath.length && forms) return form;

        // Check Inputs
        // const inputContainer = this.shadowRoot.querySelector(`#${encode(path.join("-"))}`);
        // if (inputContainer && inputs) return inputContainer.querySelector("jsonschema-input");;

        const foundInput = this.inputs[path.join(".")]; // Check Inputs
        if (foundInput && inputs) return foundInput;

        // Check Nested Form Inputs
        return form?.getFormElement(updatedPath, { forms, tables, inputs });
    };

    #requirements = {};

    attributeChangedCallback(changedProperties, oldValue, newValue) {
        super.attributeChangedCallback(changedProperties, oldValue, newValue);
        if (changedProperties === "options") this.requestUpdate();
    }

    getGlobalValue(path) {
        if (typeof path === "string") path = path.split(".");
        const resolved = this.#get(path, this.globals);
        return resolved;
    }

    // Track resolved values for the form (data only)
    updateData(localPath, value, forceUpdate = false) {
        const path = [...localPath];
        const name = path.pop();

        const reducer = (acc, key) => {
            const value = acc[key];
            return value && typeof value === "object" ? value : (acc[key] = {});
        }; // NOTE: Create nested objects if required to set a new path

        const resultParent = path.reduce(reducer, this.results);
        const resolvedParent = path.reduce(reducer, this.resolved);
        const hasUpdate = resolvedParent[name] !== value;

        const globalValue = this.getGlobalValue(localPath);

        // NOTE: Forms with nested forms will handle their own state updates
        if (this.isUndefined(value)) {
            // Continue to resolve and re-render...
            if (globalValue) {
                value = resolvedParent[name] = globalValue;
                const input = this.getFormElement(localPath);
                if (input) {
                    input.updateData(globalValue);
                    this.onOverride(name, globalValue, path);
                }
            } else resolvedParent[name] = undefined;

            resultParent[name] = undefined; // NOTE: Will be removed when stringified
        } else {
            resultParent[name] = value === globalValue ? undefined : value; // Retain association with global value
            resolvedParent[name] =
                isObject(value) && isObject(resolvedParent) ? merge(value, resolvedParent[name]) : value; // Merge with existing resolved values
        }

        if (hasUpdate || forceUpdate) this.onUpdate(localPath, value); // Ensure the value has actually changed
    }

    #addMessage = (name, message, type) => {
        if (Array.isArray(name)) name = name.join("-"); // Convert array to string
        const container = this.shadowRoot.querySelector(`#${encode(name)} .${type}`);
        const item = new InspectorListItem(message);
        container.appendChild(item);
    };

    #clearMessages = (localPath, type) => {
        if (Array.isArray(localPath)) localPath = localPath.join("-"); // Convert array to string

        if (!localPath.length) return;

        const container = this.shadowRoot.querySelector(`#${encode(localPath)} .${type}`);

        if (container) {
            const nChildren = container.children.length;
            container.innerHTML = "";

            // Track errors and warnings
            if (type === "errors") this.#nErrors -= nChildren;
            if (type === "warnings") this.#nWarnings -= nChildren;
        }
    };

    status;
    checkStatus = () => {
        return checkStatus.call(this, this.nWarnings, this.nErrors);
    };

    throw = (message) => {
        this.onThrow(message, this.identifier);
        throw new Error(message);
    };

    #contextMenus = [];

    validateSchema = (resolved, schema, name) => {
        return validator
            .validate(resolved, schema)
            .errors.map((e) => {
                const propName = e.path.slice(-1)[0] ?? name ?? (e.property === "instance" ? "Form" : e.property);
                const rowName = e.path.slice(-2)[0];

                const isRow = typeof rowName === "number";

                const resolvedValue = e.path.reduce((acc, token) => acc[token], resolved);

                // ------------ Exclude Certain Errors ------------

                // Allow for constructing types from object types
                if (e.message.includes("is not of a type(s)") && "properties" in schema && schema.type === "string")
                    return;

                // Ignore required errors if value is empty
                if (e.name === "required" && !this.validateEmptyValues && !(e.property in e.instance)) return;

                // Non-Strict Rule
                if (schema.strict === false && e.message.includes("is not one of enum values")) return;

                // Allow referring to floats as null (i.e. JSON NaN representation)
                if (e.message === "is not of a type(s) number") {
                    if (resolvedValue === "NaN") return;
                    else if (resolvedValue === null) {
                    } else if (isRow) e.message = `${e.message}. ${templateNaNMessage}`;
                }

                const prevHeader = name ? header(name) : "Row";

                return {
                    type: "error",
                    message: `${
                        typeof propName === "string"
                            ? `${header(propName)}${isRow ? ` on ${prevHeader} ${rowName}` : ""}`
                            : `${prevHeader} ${propName}`
                    } ${e.message}.`,
                };
            })
            .filter((v) => !!v);
    };

    validate = async (resolved = this.resolved) => {
        // Validate against the entire JSON Schema
        const copy = structuredClone(resolved);
        delete copy.__disabled;

        const result = this.validateSchema(copy, this.schema);

        const resolvedErrors = this.#resolveErrors(result, this.base, resolved);

        // Check if any required inputs are missing
        const requiredButNotSpecified = await this.#validateRequirements(resolved); // get missing required paths
        const isValid = !requiredButNotSpecified.length;

        // Check if all inputs are valid
        const flaggedInputs = this.shadowRoot ? this.shadowRoot.querySelectorAll(".invalid") : [];

        if (resolvedErrors.length) {
            const len = resolvedErrors.length;
            if (len === 1) this.throw(resolvedErrors[0].message);
            else this.throw(`${len} JSON Schema errors detected.`);
        }

        const allErrors = Array.from(flaggedInputs)
            .map((inputElement) => {
                return Array.from(inputElement.nextElementSibling.children).map((li) => li.message);
            })
            .flat();

        const nMissingRequired = allErrors.reduce((acc, curr) => {
            return (acc += curr.includes(this.#isARequiredPropertyString) ? 1 : 0);
        }, 0);

        // Print out a detailed error message if any inputs are missing
        let message = isValid
            ? ""
            : requiredButNotSpecified.length === 1
              ? `<b>${header(requiredButNotSpecified[0])}</b> is not defined`
              : `${requiredButNotSpecified.length} required inputs are not specified properly`;
        if (requiredButNotSpecified.length !== nMissingRequired)
            console.warn("Disagreement about the correct error to throw...");

        // if (!isValid && allErrors.length && nMissingRequired === allErrors.length) message = `${nMissingRequired} required inputs are not defined.`;

        // Check if all inputs are valid
        if (flaggedInputs.length) {
            flaggedInputs[0].focus();
            if (!message) {
                if (flaggedInputs.length === 1) {
                    const path = flaggedInputs[0].path;
                    const schema = this.getSchema(path);
                    message = `<b>${header(schema.title ?? path.join("."))}</b> is not valid`;
                } else message = `${flaggedInputs.length} invalid form values`;
            }
            message += `${
                this.base.length ? ` in the <b>${this.base.join(".")}</b> section` : ""
            }. Please check the highlighted fields.`;
        }

        if (message) this.throw(message);

        // Validate nested forms (skip disabled)
        for (let name in this.forms) {
            const tab = this.tabs[name];
            if (!tab || !tab.disabled) await this.forms[name].validate(resolved ? resolved[name] : undefined); // Validate nested forms too
        }

        for (let key in this.tables) {
            try {
                this.tables[key].validate(resolved ? resolved[key] : undefined); // Validate nested tables too
            } catch (error) {
                const title = this.tables[key].schema.title;
                const message = error.message.replace(
                    "this table",
                    `the <b>${header(title ?? [...this.base, key].join("."))}</b> table`
                );
                this.throw(message);
                break;
            }
        }

        return true;
    };

    #get = (path, object = this.resolved, omitted = [], skipped = []) => {
        // path = path.slice(this.base.length); // Correct for base path
        if (!path) throw new Error("Path not specified");
        return path.reduce((acc, curr, i) => {
            const tempAcc = acc?.[curr] ?? acc?.[omitted.find((str) => acc[str] && acc[str][curr])]?.[curr];
            if (tempAcc) return tempAcc;
            else {
                const level1 = acc?.[skipped.find((str) => acc[str])];
                if (level1) {
                    // Handle items-like objects
                    const result = this.#get(path.slice(i), level1, omitted, skipped);
                    if (result) return result;

                    // Handle pattern properties objects
                    const got = Object.keys(level1).find((key) => {
                        const result = this.#get(path.slice(i + 1), level1[key], omitted, skipped);
                        if (result && typeof result === "object") return result; // Schema are objects...
                    });

                    if (got) return level1[got];
                }
            }
        }, object);
    };

    #checkRequiredAfterChange = async (localPath) => {
        const path = [...localPath];
        const name = path.pop();
        const isValid = await this.triggerValidation(name, path, false);
        if (!isValid) return true;
    };

    // Resolve all references on the schema when set (INTERNAL USE ONLY)
    #schema;

    set schema(schema) {
        this.#schema = schema;
        this.#schema = replaceRefsWithValue(schema);
    }

    get schema() {
        return this.#schema;
    }

    getSchema(path, schema = this.schema) {
        if (typeof path === "string") path = path.split(".");

        // NOTE: Still must correct for the base here
        if (this.base.length) {
            const base = this.base.slice(-1)[0];
            const indexOf = path.indexOf(base);
            if (indexOf !== -1) path = path.slice(indexOf + 1);
        }

        // NOTE: Refs are now pre-resolved
        const resolved = this.#get(path, schema, ["properties", "patternProperties"], ["patternProperties", "items"]);
        // if (resolved?.["$ref"]) return this.getSchema(resolved["$ref"].split("/").slice(1)); // NOTE: This assumes reference to the root of the schema

        return resolved;
    }

    #renderInteractiveElement = (name, info, required, path = [], value, propertyType) => {
        let isRequired = this.#isRequired([...path, name]);

        const localPath = [...path, name];
        const externalPath = [...this.base, ...localPath];

        const resolved = this.#get(path, this.resolved);
        if (value === undefined) value = resolved[name];

        const isConditional = this.#getLink(externalPath) || typeof isRequired === "function"; // Check the two possible ways of determining if a field is conditional

        if (isConditional && !isRequired)
            isRequired = required[name] = async () => {
                const isRequiredAfterChange = await this.#checkRequiredAfterChange(localPath);
                if (isRequiredAfterChange) {
                    return true;
                } else {
                    const linkResults = await this.#applyToLinkedProperties(this.#checkRequiredAfterChange, localPath); // Check links
                    if (linkResults.includes(true)) return true;
                    // Handle updates when no longer required
                    else return false;
                }
            };

        const interactiveInput = new JSONSchemaInput({
            schema: info,
            path: localPath,
            value,
            form: this,
            controls: this.controls[name],
            required: isRequired,
            conditional: isConditional,
            validateEmptyValue: this.validateEmptyValues,
            pattern: propertyType === "pattern" ? name : propertyType ?? undefined,
            renderTable: this.renderTable,
            renderCustomHTML: this.renderCustomHTML,
            showLabel: true,
        });

        this.inputs[localPath.join("-")] = interactiveInput;

        return html`
            <div id=${encode(localPath.join("-"))} class="form-section">
                ${interactiveInput}
                <div class="errors"></div>
                <div class="warnings"></div>
                <div class="info"></div>
            </div>
        `;
    };

    load = () => {
        if (!this.#loaded) Object.values(this.tables).forEach((t) => t.load());
    };

    #loaded = false;
    nLoaded = 0;

    checkAllLoaded = () => {
        const expected = [...Object.keys(this.forms), ...Object.keys(this.tables)].length;
        if (this.nLoaded === expected) {
            this.#loaded = true;
            this.onLoaded();
        }
    };

    // willValidateWhenEmpty = (k) =>  (Array.isArray(this.validateEmptyValues) && this.validateEmptyValues.includes(k)) || this.validateEmptyValues;

    #validateRequirements = async (resolved = this.resolved, requirements = this.#requirements, parentPath) => {
        let invalid = [];

        for (let name in requirements) {
            let isRequired = this.#isRequired(name, requirements);

            if (this.tabs[name]?.disabled) continue; // Skip disabled tabs

            // // NOTE: Uncomment to block checking requirements inside optional properties
            // if (!requirements[name][selfRequiredSymbol] && !resolved[name]) continue; // Do not continue checking requirements if absent and not required

            if (typeof isRequired === "function") isRequired = await isRequired.call(this.resolved);
            if (isRequired) {
                let path = parentPath ? `${parentPath}-${name}` : name;

                // if (typeof isRequired === "object" && !Array.isArray(isRequired))
                //     invalid.push(...(await this.#validateRequirements(resolved[name], isRequired, path)));
                // else
                if (this.isUndefined(resolved[name]) && this.validateEmptyValues) invalid.push(path);
            }
        }

        return invalid;
    };

    // Checks missing required properties and throws an error if any are found
    onInvalid = () => {};
    onLoaded = () => {};
    onUpdate = () => {};
    onOverride = () => {};

    // #deleteExtraneousResults = (results, schema) => {
    //     for (let name in results) {
    //         if (!schema.properties || !(name in schema.properties)) delete results[name];
    //         else if (results[name] && typeof results[name] === "object" && !Array.isArray(results[name]))
    //             this.#deleteExtraneousResults(results[name], schema.properties[name]);
    //     }
    // };

    #getRenderable = (schema = {}, required, ignore = {}, path, recursive = false) => {
        const entries = Object.entries(schema.properties ?? {});

        const isArrayOfArrays = (arr) => !!arr.find((v) => Array.isArray(v));

        const flattenRecursedValues = (arr) => {
            const newArr = [];
            arr.forEach((item) => {
                if (isArrayOfArrays(item)) newArr.push(...item);
                else newArr.push(item);
            });

            return newArr;
        };

        const isRenderable = (key, value) => {
            if (recursive && value.properties)
                return this.#getRenderable(
                    value,
                    this.#isRequired([...path, key]),
                    getIgnore(ignore, key),
                    [...path, key],
                    true
                );
            else return [key, value];
        };

        const res = entries
            .map(([key, value]) => {
                if (!value.properties && key === "definitions") return false; // Skip definitions
                if (this.ignore["*"]?.[key])
                    return false; // Skip all properties with this name
                else if (this.ignore[key] === true) return false; // Skip this property
                if (this.showLevelOverride >= path.length) return isRenderable(key, value);
                if (required[key]) return isRenderable(key, value);
                if (this.#getLink([...this.base, ...path, key])) return isRenderable(key, value);
                if (!this.onlyRequired) return isRenderable(key, value);
                return false;
            })
            .filter((result) => !!result);

        return flattenRecursedValues(res); // Flatten on the last pass
    };

    validateOnChange = () => {};
    onStatusChange = () => {};
    onStatusUpdate = () => {};
    onThrow = () => {};
    renderTable = () => {};
    renderCustomHTML = () => {};

    #getLink = (args) => {
        if (typeof args === "string") args = args.split("-");
        const group = this.#getGroup(args);
        if (!group) return;
        return group.validate ? group : undefined;
    };

    #getGroup = (args) => {
        if (typeof args === "string") args = args.split("-");
        const group = this.groups.find((linked) => linked.properties.find((link) => link.join("-") === args.join("-")));
        return group;
    };

    #applyToLinkedProperties = (fn, externalPath) => {
        const links = this.#getLink(externalPath)?.properties;
        if (!links) return [];
        return Promise.all(
            links
                .map((link) => {
                    const linkEl = this.shadowRoot.getElementById(`${link.join("-")}`);
                    return fn(link, linkEl);
                })
                .flat()
        );
    };

    // Check if all links are not required anymore
    #isLinkResolved = async (pathArr) => {
        return (
            await this.#applyToLinkedProperties((link) => {
                const isRequired = this.#isRequired(link.slice((this.base ?? []).length));
                if (typeof isRequired === "function") return !isRequired.call(this.resolved);
                else return !isRequired;
            }, pathArr)
        ).reduce((a, b) => a && b, true);
    };

    #isRequired = (path, requirements = this.#requirements) => {
        if (typeof path === "string") path = path.split("-");
        // path = path.slice(this.base.length); // Remove base path
        let res = path.reduce((obj, key) => obj?.[key], requirements);
        if (typeof res === "object") res = res[selfRequiredSymbol];
        return res;
    };

    #getGroupElement = (externalPath) => {
        const link = this.#getGroup(externalPath);
        if (!link) return;
        return this.shadowRoot.querySelector(`[data-name="${link.name}"]`);
    };

    isUndefined(value) {
        return value === undefined || value === "";
    }

    #isARequiredPropertyString = `is a required property`;

    #resolveErrors = (errors, externalPath, parent) => {
        return errors
            .map((e) => {
                // Custom Error Transformations
                if (this.transformErrors) {
                    const name = externalPath.slice(-1)[0];
                    const res = this.transformErrors(e, externalPath, parent[name]);
                    if (res === false) return;
                }

                return e;
            })
            .filter((v) => !!v);
    };

    // Assume this is going to return as a Promise—even if the change function isn't returning one
    triggerValidation = async (name, path = [], checkLinks = true, input, schema, parent, hooks = {}) => {
        const { onError, onWarning, onInfo } = hooks;

        const localPath = [...path, name].filter((str) => typeof str === "string"); // Ignore row information
        const externalPath = [...this.base, ...localPath];
        const pathToValidate = [...this.base, ...path];

        const undefinedPathToken = localPath.findIndex((str) => !str && typeof str !== "number") !== -1;
        if (undefinedPathToken) return true; // Will be unable to get schema anyways (additionalProperties)

        if (!input) input = this.getFormElement(localPath, { inputs: true });
        if (!parent) parent = this.#get(path, this.resolved);
        if (!schema) schema = this.getSchema(localPath);

        const value = parent[name];

        const skipValidation = !this.validateEmptyValues && value === undefined;
        const validateArgs = input.pattern || skipValidation ? [] : [value, schema];

        const jsonSchemaErrors = validateArgs.length === 2 ? this.validateSchema(...validateArgs, name) : [];

        const valid = skipValidation ? true : await this.validateOnChange(name, parent, pathToValidate, value);

        if (valid === null) return null; // Skip validation / data change if the value is null

        const isRequired = this.#isRequired(localPath) || (!input.table && input.required); // Do not trust required status of table validations

        let warnings = Array.isArray(valid)
            ? valid.filter((info) => info.type === "warning" && (!isRequired || !info.missing))
            : [];

        const errors = [
            ...(Array.isArray(valid)
                ? valid?.filter((info) => info.type === "error" || (isRequired && info.missing))
                : []), // Derived Errors
            ...jsonSchemaErrors, // JSON Schema Errors
        ];

        const info = Array.isArray(valid) ? valid?.filter((info) => info.type === "info") : [];

        const isUndefined = this.isUndefined(parent[name]);

        const hasLinks = this.#getLink(externalPath);
        if (hasLinks) {
            if (checkLinks) {
                if (!(await this.#isLinkResolved(externalPath))) {
                    errors.push(...warnings); // Move warnings to errors if the element is linked
                    warnings = [];

                    // Clear old errors and warnings on linked properties
                    this.#applyToLinkedProperties((path) => {
                        const internalPath = path.slice((this.base ?? []).length);
                        this.#clearMessages(internalPath, "errors");
                        this.#clearMessages(internalPath, "warnings");
                    }, externalPath);
                }
            }
        }

        if (!errors.length) {
            if (isUndefined) {
                // Throw at least a basic warning if a non-linked property is required and missing
                if (!hasLinks && isRequired) {
                    if (this.validateEmptyValues) {
                        const rowName = pathToValidate.slice(-1)[0];
                        const isRow = typeof rowName === "number";

                        errors.push({
                            message: `${schema.title ?? header(name)} ${this.#isARequiredPropertyString}. ${
                                schema.type === "number"
                                    ? isRow
                                        ? templateNaNMessage
                                        : "<br><small>Use the 'I Don't Know' checkbox if unsure.</small>"
                                    : ""
                            }`,
                            type: "error",
                            missing: true,
                        });
                    } else {
                        warnings.push({
                            message: `${schema.title ?? header(name)} is a suggested property.`,
                            type: "warning",
                            missing: true,
                        });
                    }
                }
            }

            // Validate Regex Pattern automatically
            else if (schema.pattern) {
                const regex = new RegExp(schema.pattern, schema.flags);
                if (!regex.test(parent[name])) {
                    errors.push({
                        message: `${schema.title ?? header(name)} does not match the required pattern (${regex}).`,
                        type: "error",
                    });
                }
            }
        }

        // Clear old errors and warnings
        this.#clearMessages(localPath, "errors");
        this.#clearMessages(localPath, "warnings");
        this.#clearMessages(localPath, "info");

        const isFunction = typeof valid === "function";
        const isValid =
            valid === true ||
            valid == undefined ||
            isFunction ||
            (Array.isArray(valid) && !valid.find((error) => error.type === "error"));

        if (!isValid && errors.length === 0) errors.push({ type: "error", message: "Invalid value detected" });

        const resolvedErrors = this.#resolveErrors(errors, externalPath, parent);

        // Track errors and warnings
        const updatedWarnings = warnings.map((info) => (onWarning ? onWarning(info) : info)).filter((v) => !!v);
        const updatedErrors = resolvedErrors.map((info) => (onError ? onError(info) : info)).filter((v) => !!v);

        this.#nErrors += updatedErrors.length;
        this.#nWarnings += updatedWarnings.length;
        this.checkStatus();

        // Show aggregated errors and warnings (if any)
        updatedWarnings.forEach((info) => (onWarning ? "" : this.#addMessage(localPath, info, "warnings")));
        info.forEach((info) => (onInfo ? onInfo(info) : this.#addMessage(localPath, info, "info")));

        const groupEl = this.#getGroupElement(externalPath);

        if (groupEl) {
            groupEl.classList[resolvedErrors.length ? "add" : "remove"]("error");
            groupEl.classList[warnings.length ? "add" : "remove"]("warning");
        }

        if (isValid && updatedErrors.length === 0) {
            input.classList.remove("invalid");

            await this.#applyToLinkedProperties((path, element) => {
                element.classList.remove("required", "conditional"); // Links manage their own error and validity states, but only one needs to be valid
            }, localPath);

            if (isFunction) valid(); // Run if returned value is a function

            return true;
        } else {
            // Add new invalid classes and errors
            input.classList.add("invalid");

            // Only add the conditional class for linked elements
            await this.#applyToLinkedProperties(
                (name, element) => element.classList.add("required", "conditional"),
                [...path, name]
            );

            updatedErrors.forEach((info) => (onError ? "" : this.#addMessage(localPath, info, "errors")));
            // element.title = errors.map((info) => info.message).join("\n"); // Set all errors to show on hover

            return false;
        }
    };

    tabs = {};

    #render = (schema, results, required = {}, ignore = {}, path = []) => {
        let isLink = Symbol("isLink");

        const hasPatternProperties = !!schema.patternProperties;
        const allowAdditionalProperties = schema.additionalProperties !== false;

        // Filter non-required properties (if specified) and render the sub-schema
        const renderable = this.#getRenderable(schema, required, ignore, path);
        // // Filter non-required properties (if specified) and render the sub-schema
        // const renderable = path.length ? this.#getRenderable(schema, required) : Object.entries(schema.properties ?? {})

        const hasProperties = renderable.length > 0 || hasPatternProperties || allowAdditionalProperties;

        if (!hasProperties) return html`<div id="empty">${this.emptyMessage}</div>`;
        let renderableWithLinks = renderable.reduce((acc, [name, info]) => {
            const externalPath = [...this.base, ...path, name];
            const link = this.#getGroup(externalPath); // Use the base path to find a link
            if (link) {
                if (!acc.find(([_, info]) => info === link)) {
                    const entry = [link.name, link];
                    entry[isLink] = true;
                    acc.push(entry);
                }
            } else acc.push([name, info]);

            return acc;
        }, []);

        const getRequiredValue = (name) => {
            const value = required[name];
            return value && typeof value === "object" ? value[selfRequiredSymbol] : value;
        };

        const sorted = renderableWithLinks

            // Sort alphabetically
            .sort(([name], [name2]) => {
                const header1 = header(name);
                const header2 = header(name2);
                if (header1.toLowerCase() < header2.toLowerCase()) {
                    return -1;
                }
                if (header1.toLowerCase() > header2.toLowerCase()) {
                    return 1;
                }
                return 0;
            })

            // Sort required properties to the top
            .sort((e1, e2) => {
                const [name] = e1;
                const [name2] = e2;

                if (getRequiredValue(name) && !getRequiredValue(name2)) return -1; // first required
                if (!getRequiredValue(name) && getRequiredValue(name2)) return 1; // second required

                if (e1[isLink] && !e2[isLink]) return -1; // first link
                if (!e1[isLink] && e2[isLink]) return 1; // second link

                return 0; // Both required
            })

            // Prioritise properties without other properties (e.g. name over NWBFile)
            .sort((e1, e2) => {
                const [_, info] = e1;
                const [__, info2] = e2;

                if (e1[isLink] || e2[isLink]) return 0;

                if (info2.properties && info.properties) return 0;
                else if (info2.properties) return -1;
                else if (info.properties) return 1;
                else return 0;
            });

        if (schema.order) {
            sorted.sort(([name], [name2]) => {
                const index = schema.order.indexOf(name);
                const index2 = schema.order.indexOf(name2);
                if (index === -1) return 1;
                if (index2 === -1) return -1;
                return index - index2;
            });
        }

        const finalSort = this.sort ? sorted.sort(this.sort) : sorted;

        const tabItems = [];

        let rendered = finalSort.map((entry) => {
            const [name, info] = entry;

            // const hasPatternProperties = !!info.patternProperties;

            // Render linked properties
            if (entry[isLink]) {
                const linkedProperties = info.properties.map((path) => {
                    const pathCopy = [...path].slice((this.base ?? []).length);
                    const name = pathCopy.pop();
                    return this.#renderInteractiveElement(name, schema.properties[name], required, pathCopy);
                });
                return html`
                    <div class="link" data-name="${info.name}">
                        <div>${linkedProperties}</div>
                    </div>
                `;
            }

            // ------------------------- Directly render the interactive property element -------------------------
            if (!info.properties) return this.#renderInteractiveElement(name, info, required, path);

            // ------------------------- Create tabs -------------------------
            const localPath = [...path, name];

            // Check properties that will be rendered before creating the accordion
            const base = [...this.base, ...localPath];

            const explicitlyRequired = schema.required?.includes(name) ?? false;

            const headerName = header(info.title ?? name);

            const renderableInside = this.#getRenderable(info, required[name], ignore, localPath, true);

            const __disabled = this.results.__disabled ?? (this.results.__disabled = {});
            const __interacted = __disabled.__interacted ?? (__disabled.__interacted = {});

            const hasInteraction = __interacted[name]; // NOTE: This locks the specific value to what the user has chosen...

            const { __disabled: __tempDisabledGlobal = {} } = this.getGlobalValue(localPath.slice(0, -1));

            const __disabledGlobal = structuredClone(__tempDisabledGlobal); // NOTE: Cloning ensures no property transfer

            let isGlobalEffect = !hasInteraction || (!hasInteraction && __disabledGlobal.__interacted?.[name]); // Indicate whether global effect is used

            const __disabledResolved = isGlobalEffect ? __disabledGlobal : __disabled;

            const isDisabled = !!__disabledResolved[name];

            const nestedResults = __disabled[name] ?? results[name] ?? this.results[name]; // One or the other will exist—depending on global or local disabling

            if (renderableInside.length) {
                const ignore = getIgnore(this.ignore, name);

                const ogContext = this;
                const nested = (this.forms[name] = new JSONSchemaForm({
                    identifier: this.identifier,
                    schema: info,
                    results: { ...nestedResults },
                    globals: this.globals?.[name],

                    controls: this.controls[name],

                    onUpdate: (internalPath, value, forceUpdate) => {
                        const path = [...localPath, ...internalPath];
                        this.updateData(path, value, forceUpdate);
                    },

                    transformErrors: this.transformErrors,

                    required: required[name], // Scoped to the sub-schema
                    ignore,
                    dialogOptions: this.dialogOptions,
                    dialogType: this.dialogType,
                    onlyRequired: this.onlyRequired,
                    showLevelOverride: this.showLevelOverride,
                    deferLoading: this.deferLoading,
                    groups: this.groups,
                    validateOnChange: (...args) => this.validateOnChange(...args),
                    onThrow: (...args) => this.onThrow(...args),
                    validateEmptyValues: this.validateEmptyValues,
                    onStatusUpdate: ({ errors, warnings }) => (tabItem.status = { errors, warnings }),
                    onStatusChange: () => {
                        this.checkStatus(); // Forward status changes to the parent form
                    },
                    onInvalid: (...args) => this.onInvalid(...args),
                    onLoaded: () => {
                        this.nLoaded++;
                        this.checkAllLoaded();
                    },
                    renderCustomHTML: function (...args) {
                        return ogContext.renderCustomHTML.call(this, ...args);
                    },
                    renderTable: function (...args) {
                        return ogContext.renderTable.call(this, ...args);
                    },
                    onOverride: (...args) => this.onOverride(...args),
                    base,
                }));
            }

            const disabledPath = [...path, "__disabled"];
            const interactedPath = [...disabledPath, "__interacted"];

            const tabItem = (this.tabs[name] = new TabItem({
                name: headerName,
                content: this.forms[name],
                disabled: isDisabled,
            }));

            tabItem.id = name; // assign name to accordion id

            const isOptionalProperty = !explicitlyRequired;

            if (isOptionalProperty) {
                const context = new ContextMenu({
                    target: tabItem,
                    items: [
                        {
                            id: "enable-property",
                            label: "Enable",
                            disabled: !isDisabled,
                            onclick: (path) => {
                                isGlobalEffect = false;
                                enable();
                                this.updateData([...interactedPath, name], true, true);
                                this.onUpdate(localPath, this.results[name]);
                            },
                        },
                        {
                            id: "disable-property",
                            label: "Disable",
                            disabled: isDisabled,
                            onclick: (path) => {
                                isGlobalEffect = false;
                                disable();
                                this.updateData([...interactedPath, name], true, true);
                                this.onUpdate(localPath, this.results[name]);
                            },
                        },
                    ],
                });

                this.#contextMenus.push(context);
                document.body.append(context);

                const disable = () => {
                    tabItem.disabled = true;

                    const enableButton = context.shadowRoot.querySelector("#enable-property");
                    const disableButton = context.shadowRoot.querySelector("#disable-property");
                    if (disableButton) disableButton.setAttribute("disabled", "");
                    if (enableButton) enableButton.removeAttribute("disabled");

                    const target = this.results;
                    const value = target[name] ?? {};

                    let update = true;
                    if (target.__disabled?.[name] && isGlobalEffect) update = false;

                    // Disabled path is set to actual value
                    if (update) this.updateData([...disabledPath, name], value);

                    // Actual data is set to undefined
                    this.updateData(localPath, undefined);

                    this.checkStatus();
                };

                const enable = () => {
                    tabItem.disabled = false;

                    const enableButton = context.shadowRoot.querySelector("#enable-property");
                    const disableButton = context.shadowRoot.querySelector("#disable-property");
                    if (enableButton) enableButton.setAttribute("disabled", "");
                    if (disableButton) disableButton.removeAttribute("disabled");

                    const { __disabled = {} } = this.results;

                    // Actual value is restored to the cached value
                    if (__disabled[name]) this.updateData(localPath, __disabled[name]);

                    // Cached value is cleared
                    this.updateData([...disabledPath, name], undefined);

                    this.checkStatus();
                };

                if (isGlobalEffect) isDisabled ? disable() : enable();
            }

            tabItems.push(tabItem);
            return "";
        });

        if (hasPatternProperties) {
            const patternProps = Object.entries(schema.patternProperties).map(([key, schema]) => {
                return this.#renderInteractiveElement(
                    key,
                    {
                        ...schema,
                        title: `Pattern Properties <small><small style="font-weight: normal">${key}</small></small>`,
                    },
                    required,
                    path,
                    results,
                    "pattern"
                );
            });

            rendered = [...rendered, ...patternProps];
        }

        if (tabItems.length) {
            const tabs = new Tabs({ items: tabItems, contentPadding: "25px" });
            rendered.push(tabs);
        }

        const additionalProps = getEditableItems(results, additionalPropPattern, { schema });

        // Render additional properties
        if (allowAdditionalProperties) {
            // NOTE: If no pre-existing additional properties exist, exclude the entire rendering group
            if (!additionalProps.length) return rendered;

            const additionalElement = this.#renderInteractiveElement(
                "",
                {
                    title: `Additional Properties`,
                    ...schema,
                },
                required,
                path,
                results,
                additionalPropPattern
            );

            return [...rendered, additionalElement];
        }

        // Delete additional properties off the final results
        else {
            additionalProps.forEach(({ key }) => {
                delete results[key];
            });
        }

        return rendered;
    };

    #registerRequirements = (schema, requirements = {}, acc = this.#requirements, path = []) => {
        if (!schema) return;

        const isItem = (schema) => schema.items && schema.items.properties;
        if (isItem(schema)) schema = schema.items;

        if (schema.required) schema.required.forEach((key) => (acc[key] = true));

        for (let key in requirements) acc[key] = requirements[key]; // Overwrite standard requirements with custom requirements

        if (schema.properties) {
            Object.entries(schema.properties).forEach(([key, value]) => {
                const isPropItem = isItem(value);

                if (value.properties || isPropItem) {
                    const fullPath = [...path, key];
                    let nextAccumulator = acc[key];
                    const isNotObject = typeof nextAccumulator !== "object";
                    if (!nextAccumulator || isNotObject)
                        nextAccumulator = acc[key] = { [selfRequiredSymbol]: !!(nextAccumulator && !isPropItem) };
                    this.#registerRequirements(value, requirements[key], nextAccumulator, fullPath);
                }
            });
        }
    };

    updated() {
        this.checkAllLoaded(); // Throw if no tables
        this.#toggleRendered(); // Toggle internal render state
    }

    #resetLoadState() {
        this.#loaded = false;
        this.nLoaded = 0;
        this.inputs = {};
    }

    // Check if everything is internally rendered
    get rendered() {
        const isRendered = resolve(this.#rendered, () =>
            Promise.all([...Object.values(this.forms), ...Object.values(this.tables)].map(({ rendered }) => rendered))
        );
        return isRendered;
    }

    #resetContextMenus = () => {
        this.#contextMenus.forEach((o) => o.remove());
        this.#contextMenus = [];
    };

    disconnectedCallback() {
        super.disconnectedCallback();
        this.#resetContextMenus();
    }

    render() {
        this.#updateRendered(); // Create a new promise to check on the rendered state

        // Remove context menus
        this.#resetContextMenus();

        this.#resetLoadState();

        const schema = this.schema ?? {};

        this.resolved = structuredClone(this.results); // Track resolved values as a copy of the user-specified results

        // Register default properties
        resolveProperties(schema.properties, this.resolved, this.globals);

        // // Delete extraneous results
        // this.#deleteExtraneousResults(this.results, this.schema);

        this.#registerRequirements(this.schema, this.required);

        return html`
            ${schema.description
                ? html`<h4>Description</h4>
                      <p class="guided--text-input-instructions">${unsafeHTML(schema.description)}</p>`
                : ""}
            ${this.#render(schema, this.resolved, this.#requirements, this.ignore)}
        `;
    }
}

customElements.get("nwb-jsonschema-form") || customElements.define("nwb-jsonschema-form", JSONSchemaForm);

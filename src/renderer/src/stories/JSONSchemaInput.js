import { LitElement, html } from "lit";
import { FilesystemSelector } from "./FileSystemSelector";

import { BasicTable } from "./BasicTable";
import { header } from "./forms/utils";

import { Button } from "./Button";
import { List } from './List'
import { Modal } from "./Modal";

const filesystemQueries = [ "file", "directory" ];

export class JSONSchemaInput extends LitElement {
    
    constructor(props){
        super()

        const { 
            info, 
            parent, 
            path,
    
            form, 
            
            validateOnChange = true ,
            required = false
        } = props

        this.info = info
        this.parent = parent
        this.path = path

        this.form = form

        this.validateOnChange = validateOnChange
        this.required = required

        if (props.value) this.value = props.value
    }

    render(){

            const { validateOnChange, info, parent, path } = this

            const name = ((typeof path === 'string') ? path.split('-') : path).slice(-1)[0]            

            const isArray = info.type === "array"; // Handle string (and related) formats / types

            const hasItemsRef = "items" in info && "$ref" in info.items;
            if (!("items" in info) || (!("type" in info.items) && !hasItemsRef)) info.items = { type: 'string' }

            if (isArray) {

                // if ('value' in this && !Array.isArray(this.value)) this.value = [ this.value ]

                const headerText = document.createElement('span')
                headerText.innerText = header(name)

                const addButton = new Button({
                    size: 'small'
                })

                addButton.innerText = 'Add Item'

                let modal;
                addButton.addEventListener('click', () => {

                    if (modal) modal.remove()

                    modal = new Modal({
                        header: headerText,
                        footer: submitButton
                    })

                    const div = document.createElement('div')
                    div.style.padding = '25px'

                    div.append(new JSONSchemaInput({ 
                        info: info.items, 
                        parent: tempParent, 
                        validateOnChange: false, 
                        path: this.path, 
                        form: this.form
                    }))

                    modal.append(div)

                    addButton.insertAdjacentElement('beforebegin', modal)

                    setTimeout(() => modal.toggle(true))
                })

                const tempParent = {}


                const list = new List({
                    items: this.value ? this.value.map(value => {return { value }}): [],
                    onChange: async () =>  {
                        parent[name] = list.items.map(o => o.value)
                        if (this.required && parent[name].length === 0) delete parent[name] // Remove if empty and required

                        await this.form.triggerValidation(name, parent, list, path)
                    }
                })

                const submitButton = new Button()
                submitButton.innerText = `Submit`
                submitButton.addEventListener('click', function() {
                    const value = tempParent[name]
                    list.add({ value })
                    modal.toggle(false)
                })
                
                return html`
                    <div>
                        ${list}
                        ${addButton}
                    </div>
                `
            }



            // Basic enumeration of properties on a select element
            if (info.enum) {
                return html`
                    <select
                        class="guided--input schema-input"
                        @input=${(ev) => this.form.updateParent(name, info.enum[ev.target.value], parent)}
                        @change=${(ev) => validateOnChange && this.form.triggerValidation(name, parent, ev.target, path)}
                    >
                        <option disabled selected value>Select an option</option>
                        ${info.enum.map(
                            (item, i) =>
                                html`<option value=${i} ?selected=${this.value === item}>${item}</option>`
                        )}
                    </select>
                `;
            } else if (info.type === "boolean") {
                return html`<input
                    type="checkbox"
                    class="schema-input"
                    @input=${(ev) => this.form.updateParent(name, ev.target.checked, parent)}
                    ?checked=${this.value ?? false}
                    @change=${(ev) => validateOnChange && this.form.triggerValidation(name, parent, ev.target, path)}
                />`;
            } else if (info.type === "string" || info.type === "number") {

                let format = info.format
                const matched = name.match(/(.+_)?(.+)_path/)
                if (!format && matched) format = (matched[2] === 'folder') ? 'directory' : matched[2]

                // Handle file and directory formats
                if (filesystemQueries.includes(format)) {
                    const el = new FilesystemSelector({
                        type: format,
                        value: this.value,
                        onSelect: (filePath) => this.form.updateParent(name, filePath, parent),
                        onChange: (filePath) => validateOnChange && this.form.triggerValidation(name, parent, el, path),
                        dialogOptions: this.form.dialogOptions,
                        dialogType: this.form.dialogType,
                    });
                    return el;
                }

                // Handle long string formats
                else if (info.format === "long" || isArray)
                    return html`<textarea
                        class="guided--input guided--text-area schema-input"
                        type="text"
                        placeholder="${info.placeholder ?? ""}"
                        style="height: 7.5em; padding-bottom: 20px"
                        maxlength="255"
                        .value="${this.value ?? ""}"
                        @input=${(ev) => {
                            this.form.updateParent(
                                name,
                                ev.target.value,
                                parent
                            );
                        }}
                        @change=${(ev) => validateOnChange && this.form.triggerValidation(name, parent, ev.target, path)}
                    ></textarea>`;
                // Handle other string formats
                else {
                    const type =
                        info.format === "date-time"
                            ? "datetime-local"
                            : info.format ?? (info.type === "string" ? "text" : info.type);
                    return html`
                        <input
                            class="guided--input schema-input"
                            type="${type}"
                            placeholder="${info.placeholder ?? ""}"
                            .value="${this.value ?? ""}"
                            @input=${(ev) => this.form.updateParent(name, ev.target.value, parent)}
                            @change=${(ev) => validateOnChange && this.form.triggerValidation(name, parent, ev.target, path)}
                        />
                    `;
                }
            }

            if (info.type === "array") {
                const itemSchema = this.form.getSchema("items", info);
                if (itemSchema.type === "object") {
                    const tableMetadata = {
                        schema: itemSchema,
                        data: this.value,
                        validateOnChange: (key, parent, v) => validateOnChange && this.form.validateOnChange(key, parent, fullPath, v),
                        onStatusChange: () => this.form.checkStatus(), // Check status on all elements
                        validateEmptyCells: this.form.validateEmptyValues,
                        deferLoading: this.form.deferLoading,
                        onLoaded: () => {
                            this.form.nLoaded++;
                            this.form.checkAllLoaded();
                        },
                        onThrow: (...args) => this.form.onThrow(...args),
                    };

                    return (this.form.tables[name] =
                        this.form.renderTable(name, tableMetadata, fullPath) || new BasicTable(tableMetadata));
                }
            }

            // Print out the immutable default value
            return html`<pre>
${info.default ? JSON.stringify(info.default, null, 2) : "No default value"}</pre
            >`;
    }
}

customElements.get("nwb-jsonschema-input") || customElements.define("nwb-jsonschema-input", JSONSchemaInput);
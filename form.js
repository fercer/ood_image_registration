// form.js

'use strict';


// Wrap in a function to avoid global scope pollution
(function(){
    const bridge_id = "batch_connect_session_context_json_bridge_tree";

    function getElementType(element) {
        // If it's a select dropdown, return 'select'
        if (element.tagName === "SELECT") return "select";

        // If it's an input, return its specific type (checkbox, radio, text, number)
        if (element.tagName === "INPUT") return element.type;

        // Fallback for textareas or generic divs
        return element.tagName.toLowerCase();
    }

    // Function to toggle visibility of widgets
    function toggle_visibility_tree(is_checked, subtree_children) {
        Object.entries(subtree_children).forEach(function(child_entry){
            let element = document.getElementById(child_entry[0]);
            let element_checked = is_checked

            if (element) {
                if (is_checked) {
                    element.closest("form > .mb-3").style.display = '';
                } else {
                    element.closest("form > .mb-3").style.display = 'none';
                }

                if (getElementType(element) === "select") {
                    toggle_visibility_selected_option(element.value.toLowerCase(), child_entry[1], is_checked);
                }
                else if (getElementType(element) == "checkbox") {
                    element_checked = element.checked
                    toggle_visibility_tree(element_checked & is_checked, child_entry[1]);
                }
            }

        });
    }

    // Function to toggle visibility of show checkboxes
    function toggle_visibility_selected_option(selected_option, subtree_children, is_visible) {
        Object.entries(subtree_children).forEach(function(option_entry){
            const is_selected = option_entry[0].endsWith(selected_option.toLowerCase());
            let child_element = document.getElementById(option_entry[0]);

            if (child_element) {
                if (is_selected & is_visible) {
                    child_element.closest("form > .mb-3").style.display = '';
                } else {
                    child_element.closest("form > .mb-3").style.display = 'none';
                } 
                toggle_visibility_tree(is_selected & is_visible & child_element.checked, option_entry[1]);
            }
        });
    }

    function add_listener(widgets_tree, is_visible, color_depth) {
        Object.entries(widgets_tree).forEach(function(widget_subtree) {
            let child_is_visible = is_visible;
            const subtree_id = widget_subtree[0];
            const subtree_children = widget_subtree[1];

            const parent_element = document.getElementById(subtree_id);
            if (parent_element) {
                if (getElementType(parent_element) === "select") {
                    parent_element.addEventListener('change', function() {
                        toggle_visibility_selected_option(this.value, subtree_children, true);
                    });
                    parent_element.closest('form > .mb-3').style.backgroundColor = "rgba(10, 10, 10, " + color_depth + ")";
                    toggle_visibility_selected_option(parent_element.value, subtree_children, child_is_visible);
                }
                else if (getElementType(parent_element) === "checkbox") {
                    parent_element.addEventListener('change', function() {
                        toggle_visibility_tree(this.checked, subtree_children);
                    });
                    parent_element.closest('form > .mb-3').style.backgroundColor = "rgba(10, 10, 10, " + color_depth + ")";
                    toggle_visibility_tree(child_is_visible & parent_element.checked, subtree_children);
                    child_is_visible = child_is_visible & parent_element.checked;
                }
                else {
                    parent_element.closest('form > .mb-3').style.backgroundColor = "rgba(10, 10, 10, " + color_depth + ")";
                }

                add_listener(subtree_children, child_is_visible, color_depth + 0.1);
            }
        });
    }

    // Event Listener: Wait for the DOM to load
    document.addEventListener("DOMContentLoaded", function() {
        const bridge_input = document.getElementById(bridge_id);

        if (bridge_input) {
            // 2. Read the value (the string we put there)
            let raw_json = bridge_input.value;

            // 3. Parse it into a Javascript Object
            try {
                const widgets_tree = JSON.parse(raw_json);
                
                console.log("Loaded Data from Bridge:", widgets_tree);

                add_listener(widgets_tree, true, 0.0);

            } catch (e) {
                console.error("Failed to parse JSON from bridge:", e);
            }
        }
    });
})();
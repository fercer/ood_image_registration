// form.js

'use strict';


// Wrap in a function to avoid global scope pollution
(function(){
    const bridge_id = "batch_connect_session_context_json_bridge";

    // Function to toggle visibility
    function toggle_visibility(is_checked, field_id) {
        let element = document.getElementById(field_id);
        
        if (element) {
            if (is_checked) {
                element.closest(".mb-3").style.display = '';
            } else {
                element.closest(".mb-3").style.display = 'none';
            }
        }
    }

    // Function to toggle visibility of show checkboxes
    function toggle_show_advanced_visibility(selected_preprocessor, preprocessors_hash) {
        Object.entries(preprocessors_hash).forEach(function(preprocessor_entry){
            const preprocessor_name = preprocessor_entry[0];
            const show_preprocessor_id = preprocessor_entry[1];

            let show_preprocessor_checkbox = document.getElementById(show_preprocessor_id);

            if (show_preprocessor_checkbox) {
                show_preprocessor_checkbox.checked = false
                show_preprocessor_checkbox.dispatchEvent(new Event('change'));
                if (preprocessor_name === selected_preprocessor) {
                    show_preprocessor_checkbox.closest(".mb-3").style.display = '';
                } else {
                    show_preprocessor_checkbox.closest(".mb-3").style.display = 'none';
                }
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
                const advanced_arguments_data = JSON.parse(raw_json);
                
                // DEBUG: Confirm it worked
                console.log("Loaded Data from Bridge:", advanced_arguments_data);

                Object.entries(advanced_arguments_data["show_checkboxes"]).forEach(function(advanced_arguments_entry) {
                    const show_preprocessor_id = advanced_arguments_entry[0];
                    const preprocessor_args_list = advanced_arguments_entry[1];

                    const show_preprocessor_checkbox = document.getElementById(show_preprocessor_id);
                    
                    preprocessor_args_list.forEach(function(argument_id) {
                        show_preprocessor_checkbox.addEventListener('change', function() {
                            toggle_visibility(this.checked, argument_id);
                        });
                        toggle_visibility(show_preprocessor_checkbox.checked, argument_id);
                    });
                });

                Object.entries(advanced_arguments_data["dropdowns"]).forEach(function(element_entry) {
                    const element_id = element_entry[0];
                    const preprocessors = element_entry[1];

                    const preprocessor_dropdown = document.getElementById(element_id);

                    preprocessor_dropdown.addEventListener('change', function() {
                        toggle_show_advanced_visibility(this.value, preprocessors)
                    });

                    toggle_show_advanced_visibility(preprocessor_dropdown.value, preprocessors)
                });
            } catch (e) {
                console.error("Failed to parse JSON from bridge:", e);
            }
        }
    });
})();
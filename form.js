// form.js

'use strict';


// Wrap in a function to avoid global scope pollution
(function(){
    const bridge_id = "batch_connect_session_context_json_bridge";

    // Function to toggle visibility
    function toggle_visibility(is_checked, widget_arguments) {
        Object.entries(widget_arguments).forEach(function(argument_entry){
            let element = document.getElementById(argument_entry[0]);
            
            if (element) {
                if (is_checked) {
                    element.closest(".mb-3").style.display = '';
                } else {
                    element.closest(".mb-3").style.display = 'none';
                }
            }
        });
    }

    // Function to toggle visibility of show checkboxes
    function toggle_show_advanced_visibility(selected_preprocessor, preprocessors_dict) {
        Object.entries(preprocessors_dict).forEach(function(preprocessor_entry){
            const preprocessor_name = preprocessor_entry[0];
            const show_preprocessor_id = preprocessor_entry[1]["widget_id"];

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
                const advanced_args_ids = JSON.parse(raw_json);
                
                // DEBUG: Confirm it worked
                console.log("Loaded Data from Bridge:", advanced_args_ids);

                Object.entries(advanced_args_ids).forEach(function(advanced_arguments_entry) {
                    const preprocessor_id = advanced_arguments_entry[0];
                    const preprocessor_opts_dict = advanced_arguments_entry[1];

                    const preprocessor_dropdown = document.getElementById(preprocessor_id);
                    preprocessor_dropdown.addEventListener('change', function() {
                        toggle_show_advanced_visibility(this.value, preprocessor_opts_dict)
                    });
                    toggle_show_advanced_visibility(preprocessor_dropdown.value, preprocessor_opts_dict)

                    Object.entries(preprocessor_opts_dict).forEach(function(opts) {
                        const show_preprocessor_id = opts[1]["widget_id"];
                        const widget_arguments = opts[1]["widget_arguments"];

                        const show_preprocessor_checkbox = document.getElementById(show_preprocessor_id);

                        if (show_preprocessor_checkbox) {
                            show_preprocessor_checkbox.addEventListener('change', function() {
                                toggle_visibility(this.checked, widget_arguments);
                            });
                            toggle_visibility(show_preprocessor_checkbox.checked, widget_arguments);
                        }
                    });
                });
            } catch (e) {
                console.error("Failed to parse JSON from bridge:", e);
            }
        }
    });
})();
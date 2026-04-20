// JavaScript widget for VideoActorExtract node
// Displays actor info JSON in a readable format

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "VideoActorExtract.widget",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VideoActorExtractor" || nodeData.name === "Video Actor Extract") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                this.addWidget("button", "View Output", "View", () => {
                    const jsonOutput = this.outputs?.[0]?.widget?.value;
                    if (jsonOutput) {
                        try {
                            const data = JSON.parse(jsonOutput);
                            let summary = "Actors found:\n";
                            for (const actor of data.actors || []) {
                                summary += `  ${actor.actor_id}: ${actor.segment_count} segment(s), ${actor.total_frames} frames\n`;
                                for (const seg of actor.segments || []) {
                                    summary += `    [${seg.start_time_sec}s - ${seg.end_time_sec}s] (${seg.frame_count} frames)\n`;
                                }
                            }
                            alert(summary);
                        } catch (e) {
                            alert("Invalid JSON output: " + e.message);
                        }
                    } else {
                        alert("No output yet. Run the workflow first.");
                    }
                });
            };
        }
    },
});

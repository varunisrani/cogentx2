<template>
    <v-app :theme="theme">
        <v-toolbar density="compact">
            <v-app-bar-nav-icon></v-app-bar-nav-icon>

            <v-toolbar-title v-if="args.command">{{ args.command }}</v-toolbar-title>
            <v-toolbar-title v-else-if="commandHistory.length >= 1">{{ commandHistory[commandHistory.length - 1].value}}</v-toolbar-title>
            <v-toolbar-title v-else>{{ currentCommand }}</v-toolbar-title>


            <v-spacer></v-spacer>

            <v-progress-circular v-if="args.is_running" color="primary" indeterminate></v-progress-circular>

            <v-btn icon>
                <v-icon v-if="args.is_running" @click="terminateProcess">mdi-stop</v-icon>
                <v-icon v-else-if="args.command" @click="runProvidedCommand">{{args.run_count == 0 ? "mdi-play" : "mdi-replay"}}</v-icon>
                <v-icon v-else-if="currentCommand" @click="rerunLastCommand">{{args.run_count == 0 ? "mdi-play" : "mdi-replay"}}</v-icon>
            </v-btn>

            <v-btn icon>
                <v-icon>mdi-dots-vertical</v-icon>
            </v-btn>
        </v-toolbar>

        <div class="terminal" flat rounded="0"
            :style="{
                height: args.height > 0 ? args.height + 'px' : 'default',
                minHeight: args.min_height > 0 ? args.min_height + 'px' : 'default',
                maxHeight: args.max_height > 0 ? args.max_height + 'px' : 'default',
                }">
            <div class="terminal-main pa-3">
                <div class="history" v-if="args.show_welcome_message">
                    <pre>{{ args.welcome_message }}</pre>
                </div>

                <div class="history" v-for="(hist, n) in args.history" :key="n">
                    <span v-if="hist.type == 'command'">{{ prompt }}</span>
                    <span v-html="terminalCodesToHtml(hist.value)"></span>
                </div>

                <!-- Interactive input area - always visible and enabled for better UX -->
                <div :class="{ input: true, grayout: args.disable_input }">
                    <div class="d-flex">
                        <span>{{ prompt }}</span>
                        <div style="width: 100%;">
                            <input ref="terminalInput" class="terminal-input" type="text" name="" id=""
                                :disabled="args.disable_input"
                                @input="updateInput"
                                @keydown="runCommand"
                                @focus="focusOnInput"
                                autocomplete="off"
                                spellcheck="false">
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </v-app>

</template>

<script>
import { ref } from "vue"
import { Streamlit } from "streamlit-component-lib"
import { useStreamlit } from "./streamlit"
import VueCommand, { createStdout } from 'vue-command'
import "vue-command/dist/vue-command.css";
import { computed } from "vue";
import { terminalCodesToHtml } from "terminal-codes-to-html";
import { da } from "vuetify/locale";

export default {
    name: "StreamlitTerminal",
    props: ["args"], // Arguments that are passed to the plugin in Python are accessible in prop "args"
    components: {
        VueCommand
    },
    setup() {
        useStreamlit() // lifecycle hooks for automatic Streamlit resize

    },

    mounted() {
        console.log("args.history", this.args.history)
        this.args.history.forEach(h => {
            this.history.push(h)
        })

        this.currentCommand = this.args.command;

        this.scrollToBottom();
    },

    watch: {
        args: {
            handler: function (val, oldVal) {
                console.log("args changed", val)
                this.commandHistory = this.args.history.filter(h => h.type == "command" && !h.not_run)
                if (this.commandHistory.length > 0) {
                    this.currentCommand = this.commandHistory[this.commandHistory.length - 1].value
                }
                this.scrollToBottom();
            },
            deep: true
        }
    },

    data: () => ({
        theme: "dark",
        prompt: ">\u00A0",
        currentCommand: "",
        // currentCommand: "asdfaaaa",
        currentInput: "",
        commandHistory: [],
        historyTraverseIdx: 0,
        history: [],
    }),

    computed: {

    },

    methods: {
        generateMsgId(date) {
            return this.args.key + "_" + date.getTime().toString() + "_" + Math.floor(Math.random() * 1000).toString()
        },
        sendMessageToStreamlit(command, args = [], kwargs = {}) {
            const date = new Date();
            const id = this.generateMsgId(date);
            const msg = { key: this.args.key, id, command, args, kwargs, date: date.toISOString() }
            console.log("Sending message to Streamlit", msg)
            Streamlit.setComponentValue(msg)
        },
        terminalCodesToHtml(code) {
            try {
                return terminalCodesToHtml(code)
            } catch (e) {
                console.warn(e)
                return code
            }
        },
        focusOnInput() {
            this.$el.querySelector('input.terminal-input').focus()
        },
        scrollToBottom() {
            this.$nextTick(() => {
                const terminalMain = this.$el.querySelector('.terminal');
                terminalMain.scrollTop = terminalMain.scrollHeight;
            });
        },
        runCommand(e) {
            if (e.key === "Enter") {
                e.preventDefault();
                const cmd = e.target.value
                this.history.push({ type: "command", value: cmd })
                console.log(cmd)
                this.$nextTick(() => {
                    const terminalMain = this.$el.querySelector('.terminal');
                    terminalMain.scrollTop = terminalMain.scrollHeight;
                });
                e.target.value = ""
                this.currentInput = ""
                this.historyTraverseIdx = 0

                this.sendMessageToStreamlit("run_command", [cmd]);
            }
            else if (e.key === "ArrowUp") {
                e.preventDefault();

                if (this.commandHistory.length === 0) {
                    return;
                }

                this.historyTraverseIdx = Math.min(this.historyTraverseIdx + 1, this.commandHistory.length);

                const historyIndex = this.commandHistory.length - this.historyTraverseIdx;
                if (historyIndex >= 0) {
                    e.target.value = this.commandHistory[historyIndex].value;
                }
            }
            else if (e.key === "ArrowDown") {
                e.preventDefault();

                if (this.historyTraverseIdx > 0) {
                    this.historyTraverseIdx = Math.max(this.historyTraverseIdx - 1, 0);
                }

                if (this.historyTraverseIdx === 0) {
                    e.target.value = this.currentInput;
                } else {
                    const historyIndex = this.commandHistory.length - this.historyTraverseIdx;
                    if (historyIndex >= 0) {
                        e.target.value = this.commandHistory[historyIndex].value;
                    }
                }
            }

            //judge ctrl +C
            else if (e.key === "c" && e.ctrlKey) {
                // judge e.target selected or not
                if (window.getSelection().toString() == "") {
                    e.preventDefault();
                    this.sendMessageToStreamlit("add_not_run_command", [this.currentInput + "^C"])
                    this.currentInput = ""
                    e.target.value = ""
                }
            }
            else {

            }

            // scroll to bottom
            // this.$nextTick(() => {
            //     const terminalMain = this.$el.querySelector('.terminal');
            //     terminalMain.scrollTop = terminalMain.scrollHeight;
            // });
        },

        updateInput(e) {
            this.currentInput = e.target.value
            console.log("this.currentInput", this.currentInput)
        },

        rerunLastCommand() {
            this.sendMessageToStreamlit("run_command", [this.currentCommand]);
        },

        runProvidedCommand() {
            this.sendMessageToStreamlit("run_command", [this.args.command]);
        },

        terminateProcess() {
            this.sendMessageToStreamlit("terminate_process");
        }
    }
}
</script>


<style lang="scss">
.terminal {
    font-size: 14px;
    font-family: 'Menlo', 'Monaco', 'Consolas', 'Courier New', monospace;
    overflow: auto;
    background-color: #1e1e1e;
    color: #f0f0f0;
    line-height: 1.5;

    .terminal-main {
        padding-bottom: 0;
    }

    .history {
        white-space: pre-wrap;
        word-break: break-all;
    }
}

.grayout {
    color: gray;
}

.terminal-input {
    outline: 0px solid transparent;
    background-color: transparent;
    color: inherit;
    width: 100%;
    border: none;
    font-family: inherit;
    font-size: inherit;
}

[contenteditable] {
    outline: 0px solid transparent;
}

/* Improve scrollbar appearance */
.terminal::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

.terminal::-webkit-scrollbar-track {
    background: #2a2a2a;
}

.terminal::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 4px;
}

.terminal::-webkit-scrollbar-thumb:hover {
    background: #777;
}
</style>
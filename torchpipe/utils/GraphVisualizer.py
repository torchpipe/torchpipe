# Copyright 2021-2023 NetEase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified from https://github.com/ray-project/ray/blob/505149c151c4fcbaf247312df9f40efde5c60ccb/python/ray/serve/experimental/gradio_visualize_graph.py
# original license: Apache-2.0 license 
# Copyright 2022 The ray Authors.

import os

_gradio = None

def lazy_import_gradio():
    global _gradio
    if _gradio is None:
        try:
            import gradio
        except ModuleNotFoundError:
            logger.error(
                "Gradio isn't installed. Run `pip install gradio` to use Gradio to "
                "visualize a Serve deployment graph."
            )
            raise

        _gradio = gradio
    return _gradio

import networkx
class Visualization:
    
    def __init__(self, configs):

        self.default_value = ""
        if isinstance(configs, str):
            assert os.path.exists(configs), f"{configs} not exists!"
            with open(configs, "r") as f:
                self.default_value = f.read()
                # self.init_value = self.default_value
        else:
            self.default_value = configs
            self.parse_configs(configs)

    def parse_configs(self, configs):
        dag = networkx.MultiDiGraph()
                    

        for key, value in configs.items():
            dag.add_node(key, size=20)
            
            if "next" in value.keys():
                nexts = value["next"]
                if isinstance(nexts, str):
                    nexts = nexts.split(",")
                    nexts = [x.strip() for x in nexts]
                for i in nexts:
                    dag.add_edge(key, i, src_key=configs[key], dst_key=configs[i])

        # different subgraph(start from different root) with different group 
        # 1. get root nodes with indegree 0
        roots = set()
        for node in dag.nodes:
            dag.nodes[node]["title"] = "\n".join(f"{key}: {value}" for key, value in configs[node].items())
            if dag.in_degree(node) == 0:
                roots.add(node)
        # 2. set group
        for i, root in enumerate(roots):
            for sub in networkx.algorithms.traversal.bfs_tree(dag, root):
                dag.nodes[sub]["group"] = i


        return dag

            

  



    def save_html(self, save_name):
        from pyvis.network import Network

        net = Network(notebook=True, cdn_resources="in_line", directed=True)
        net.from_nx(self.dag)
        net.show(save_name)


    def launch(self):
        from pyvis.network import Network
        def get_html(toml=''):
            net = Network(directed=True)

            dag = None

            error_html=''
            if isinstance(toml, dict):
                try:
                    dag = self.parse_configs(toml)
                except Exception as e:
                    error_html = str(e)
            elif toml is not None:
                try:
                    import torchpipe
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        import os
                        if not os.path.exists(tmpdir):
                            os.mkdir(tmpdir)
                        save_name = os.path.join(tmpdir, "tmp.toml")
                        with open(save_name, "w") as f:
                            f.write(toml)
                        dag = self.parse_configs(torchpipe.parse_toml(save_name))

                except Exception as e:
                    error_html = str(e)
            if not error_html:
                print(dag)
                net.from_nx(dag)
                html = net.generate_html()
                #need to remove ' from HTML
            
                html = html.replace("'", "\"")
                
                return f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; 
                display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
                allow-scripts allow-same-origin allow-popups 
                allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
                allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
            else:
                html_template = """
                            <html>
                            <body>
                                <h1>{title}</h1>
                                <p>{content}</p>
                            </body>
                            </html>
                """

                title = "Error!"
 
                error_html = html_template.format(title=title, content=error_html)

 
                return error_html

        gr = lazy_import_gradio()
        # text = gr.Textbox(value=self.default_value, lines=2, scale=3, max_lines=200)
        exmp_text = """
[jpg_decoder]\n
backend = "SyncTensor[Tensor2Mat]"\n
next = "preprocessor"\n
\n
[preprocessor]\n
"""

        with gr.Blocks() as demo:

            h = gr.HTML(value=get_html(self.default_value))
            
            default_value = self.default_value if isinstance(self.default_value,str)  else ""
            text = gr.Textbox(value=default_value, lines=2, scale=3, max_lines=200)
            self.default_value = ""
            # btn = gr.Button(value="Submit")
            # btn.click(get_html, inputs=[text], outputs=[h])

            gr.Interface(
                get_html,
                inputs=text,
                outputs=h,
                title="torchpipe graph",
                allow_flagging='never',
                examples=[[exmp_text]],
                live=True
            )
            
            # gr.Examples(
            #     examples=[[exmp_text]],
            #     inputs=text,
            #     outputs='html',
            #     fn=get_html,
            #     cache_examples=False,
            # )


        demo.launch()
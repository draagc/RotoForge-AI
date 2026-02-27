import bpy
import json
import os
from .functions import install_dependencies

deps_check = None

_SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".rotoforge_ai.json")
_PERSISTED_KEYS = ("server_mode", "server_host", "server_port", "dependencies_path")


def _load_settings(prefs):
    """Restore persisted settings from disk into addon preferences."""
    if not os.path.isfile(_SETTINGS_PATH):
        return
    try:
        with open(_SETTINGS_PATH, "r") as f:
            data = json.load(f)
        for key in _PERSISTED_KEYS:
            if key in data:
                setattr(prefs, key, data[key])
    except Exception as e:
        print(f"RotoForge AI: Could not load settings: {e}")


def _save_settings(prefs):
    """Persist current addon preferences to disk."""
    data = {key: getattr(prefs, key) for key in _PERSISTED_KEYS}
    try:
        with open(_SETTINGS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"RotoForge AI: Could not save settings: {e}")


class Install_Dependencies_Operator(bpy.types.Operator):
    """Creates a Python 3.12+ venv and installs SAM3 server dependencies (~8GB)"""
    bl_idname = "rotoforge.install_dependencies"
    bl_label = "Install dependencies"
    bl_description = "Create venv and install SAM3 dependencies (Downloads up to ~8GB)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ok = install_dependencies.install_packages()
        if ok:
            self.report({'INFO'}, 'Dependencies installed — please restart Blender to activate')
        else:
            self.report({'ERROR'}, 'Installation failed — check the system console')
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_confirm(self, event)


class Test_Dependencies_Operator(bpy.types.Operator):
    """Tests the SAM3 venv and Blender-side dependencies"""
    bl_idname = "rotoforge.test_dependencies"
    bl_label = "Check Dependencies"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        print('--- RotoForge AI: Dependencies Debug Info ---')

        debug_info = []
        blender_ok = install_dependencies.test_blender_packages()
        venv_ok = install_dependencies.test_venv()

        global deps_check

        if not blender_ok:
            debug_info.append('Issue: Blender-side packages (numpy/Pillow)')
        if not venv_ok:
            debug_info.append('Issue: SAM3 venv not ready')

        if blender_ok and venv_ok:
            debug_info.append('No issues found — ready to use')
            deps_check = 'passed'
        else:
            debug_info.append('Check the system console for details')
            deps_check = 'failed'

        def draw(self, context):
            for line in debug_info:
                self.layout.label(text=line)

        context.window_manager.popup_menu(title='Dependencies Debug Info', draw_func=draw)
        return {'FINISHED'}


class Install_Blender_Packages_Operator(bpy.types.Operator):
    """Install only the lightweight Blender-side packages (numpy, Pillow)"""
    bl_idname = "rotoforge.install_blender_packages"
    bl_label = "Install Blender packages"
    bl_description = "Install numpy and Pillow for Blender (needed for mask I/O)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ok = install_dependencies.install_blender_packages()
        if ok:
            self.report({'INFO'}, 'Blender packages installed — please restart Blender')
        else:
            self.report({'ERROR'}, 'Installation failed — check the system console')
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_confirm(self, event)


class Forceupdate_Dependencies_Operator(bpy.types.Operator):
    """Recreates the venv and reinstalls SAM3 dependencies (~8GB)"""
    bl_idname = "rotoforge.forceupdate_dependencies"
    bl_label = "Forceupdate dependencies (Redownloads ~8GB)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Stop the server first if running
        try:
            from .functions import setup_ui
            setup_ui.stop_server()
        except Exception:
            pass

        ok = install_dependencies.install_packages(force=True)
        if ok:
            self.report({'INFO'}, 'SAM3 dependencies reinstalled successfully')
        else:
            self.report({'ERROR'}, 'Reinstallation failed — check the system console')
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_confirm(self, event)


class RotoForge_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    dependencies_path: bpy.props.StringProperty(
        name="Install path",
        description="Directory where the SAM3 venv and model weights are stored (NEEDS ~8GB SPACE)",
        subtype='DIR_PATH',
        default=os.path.realpath(os.path.expanduser("~/MVD-addons dependencies/RotoForge AI"))
    )  # type: ignore

    server_mode: bpy.props.EnumProperty(
        name="Server",
        items=[
            ("local", "Local", "Launch SAM3 server as a local subprocess"),
            ("remote", "Remote", "Connect to an already-running SAM3 server on the network"),
        ],
        default="local",
    )  # type: ignore

    server_host: bpy.props.StringProperty(
        name="Host",
        description="IP address or hostname of the remote SAM3 server",
        default="",
    )  # type: ignore

    server_port: bpy.props.IntProperty(
        name="Port",
        description="Port the SAM3 server is listening on",
        default=8799,
        min=1024, max=65535,
    )  # type: ignore

    def draw(self, context):
        layout = self.layout

        # Server mode
        layout.prop(self, "server_mode", expand=True)

        if self.server_mode == "remote":
            box = layout.box()
            box.label(text="Remote Server")
            row = box.row(align=True)
            row.prop(self, "server_host")
            row.prop(self, "server_port")
            box.label(text="Run sam3_server.py on the remote machine, then enter its IP here.")

            layout.separator()
            layout.prop(self, "dependencies_path")
            try:
                blender_ok = install_dependencies.test_blender_packages()
            except Exception:
                blender_ok = False
            if blender_ok:
                layout.label(text="Blender packages (numpy, Pillow) are installed.")
            else:
                layout.label(text="Blender packages (numpy, Pillow) need to be installed:")
                layout.operator("rotoforge.install_blender_packages")
            return

        # Local mode — show install path and dependency management
        layout.prop(self, "dependencies_path")
        row = layout.split(factor=0.7)

        labels = row.column()
        operators = row.column()

        operators.operator("rotoforge.test_dependencies")

        global deps_check

        if deps_check is None:
            labels.label(text="Please check the dependencies with the button to the right:")
            return

        if deps_check == 'passed':
            labels.label(text="Dependencies are installed, nothing to do here!")
            return

        labels.label(text="Dependencies need to be installed,")
        labels.label(text="please press the button to the right:")

        install = operators.column_flow()
        install.scale_y = 2.0
        install.operator("rotoforge.install_dependencies")


classes = [
    RotoForge_Preferences,
    Install_Dependencies_Operator,
    Install_Blender_Packages_Operator,
    Forceupdate_Dependencies_Operator,
    Test_Dependencies_Operator,
]


_submodules_registered = False


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    print("RotoForge AI: Registering extension...")
    install_dependencies.register()

    # Try registering submodules that need numpy/PIL.
    # If deps aren't installed yet, these will be registered after install + reload.
    global _submodules_registered
    try:
        from .functions import data_manager
        data_manager.register()
        from .functions import setup_ui
        setup_ui.register()
        from .functions import overlay
        overlay.register()
        _submodules_registered = True
    except ImportError as e:
        print(f'RotoForge AI: Optional deps not yet installed ({e})')
        print('RotoForge AI: Install dependencies from addon preferences to enable full functionality')
    except Exception as e:
        print(f'RotoForge AI: Error during registration: {e}')


def unregister():
    print("RotoForge AI: Unregistering extension...")

    global _submodules_registered
    if _submodules_registered:
        try:
            from .functions import setup_ui
            setup_ui.stop_server()
            setup_ui.unregister()
        except Exception as e:
            print(f'RotoForge AI: Error unregistering setup_ui: {e}')

        try:
            from .functions import overlay
            overlay.unregister()
        except Exception as e:
            print(f'RotoForge AI: Error unregistering overlay: {e}')

        try:
            from .functions import data_manager
            data_manager.unregister()
        except Exception as e:
            print(f'RotoForge AI: Error unregistering data_manager: {e}')

        _submodules_registered = False

    install_dependencies.unregister()

    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

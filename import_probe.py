import os, json
from pathlib import Path
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
root=Path(__file__).resolve().parent
app=SimulationApp({'headless':True,'width':640,'height':480,'anti_aliasing':0,'renderer':'RaytracedLighting','limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
from isaacsim.core.utils.extensions import enable_extension
enable_extension('isaacsim.asset.importer.mjcf')
import omni.kit.commands, omni.usd
from pxr import UsdPhysics
ok,config=omni.kit.commands.execute('MJCFCreateImportConfig')
print('CONFIG_METHODS',dir(config),flush=True)
config.set_fix_base(False); config.set_make_default_prim(False)
if hasattr(config,'set_import_inertia_tensor'):config.set_import_inertia_tensor(True)
result=omni.kit.commands.execute('MJCFCreateAsset',mjcf_path=str(root/'model'/'compiled.xml'),import_config=config,prim_path='/World/Bike')
print('IMPORT_RESULT',result,flush=True)
stage=omni.usd.get_context().get_stage()
rows=[]
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.CollisionAPI):
        row={'path':str(prim.GetPath()),'type':prim.GetTypeName(),'apis':prim.GetAppliedSchemas()}
        if prim.HasAPI(UsdPhysics.MassAPI):
            api=UsdPhysics.MassAPI(prim);row['mass']=api.GetMassAttr().Get();row['inertia']=str(api.GetDiagonalInertiaAttr().Get())
        rows.append(row)
(root/'model'/'import_probe.json').write_text(json.dumps(rows,indent=2))
stage.GetRootLayer().Export(str(root/'model'/'imported.usda'))
print('PROBE_COMPLETE',len(rows),flush=True)
app.close()

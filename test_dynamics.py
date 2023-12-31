from dynamics import CompQuad_
from pydrake.systems.framework import LeafSystem_
from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.autodiffutils import AutoDiffXd
from pydrake.symbolic import Expression

def main():
    compquad_autodiff = CompQuad_[AutoDiffXd]

    CompQuadSystem = compquad_autodiff()
    context = CompQuadSystem.CreateDefaultContext()

    print(f'type(CompQuadSystem): {type(CompQuadSystem)}')
    print(f'type(context): {type(context)}')
    print(f'context: {context}')
    print(f'context.get_continuous_state_vector(): {context.get_continuous_state_vector()}')
    print(f'context.get_continuous_state_vector().CopyToVector(): {context.get_continuous_state_vector().CopyToVector()}')
    print(f'context.get_continuous_state_vector().CopyToVector().shape: {context.get_continuous_state_vector().CopyToVector().shape}')
    print(f'type autoDiffXd: {type(compquad_autodiff)}')
if __name__ == "__main__":
    main()
class ContextElement:
    def accept(self, visitor):
        method_name = f'visit_{self.__class__.__name__.lower()}'
        visit = getattr(visitor, method_name, None)
        if visit:
            return visit(self)
        raise NotImplementedError(f'{method_name} is not implemented in {visitor.__class__.__name__}')
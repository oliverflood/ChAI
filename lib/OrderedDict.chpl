



use List;
use Map;

record dict : serializable {
    type keyType;
    type valType;

    var table: map(keyType,valType);
    var order: list(keyType);

    proc init(in table: map(?keyType,?valType),in order: list(keyType)) {
        this.keyType = keyType;
        this.valType = valType;
        this.table = table;
        this.order = order;
    }

    proc init(table: map(?keyType,?valType)) {
        var ks = new list(keyType);
        var tbl: map(keyType,valType) = table;
        for k in tbl.keys() do
            ks.pushBack(k);
        this.init(tbl,ks);
    }

    proc init(type keyType, type valType) {
        this.keyType = keyType;
        this.valType = valType;
        this.table = new map(keyType,valType);
        this.order = new list(keyType);
    }

    proc init(entries: (?keyType,?valType) ...?n) {
        this.init(keyType,valType);
        for param i in 0..<n {
            this.insert(entries(i)[0],entries(i)[1]);
        }
    }

    proc size: int {
        const s = table.size;
        assert(s == order.size, "Table and order sizes out of sync: ", s, order.size);
        return s;
    }

    proc ref this(key: keyType) ref throws {
        if !table.contains(key) then
            throw new Error("Key not found: " + key:string);
        return table[key];
    }

    proc const this(key: keyType) const ref throws {
        if !table.contains(key) then
            throw new Error("Key not found: " + key:string);
        return table[key];
    }

    iter keys(): keyType do 
        for i in 0..<order.size do 
            yield order[i];

    iter values(): valType do 
        for k in keys() do 
            yield table[k];

    iter these() do
        for k in order {
            yield (k,table[k]);
        }

    proc ref insert(in key: keyType, in value: valType) {
        if !order.contains(key) then
            order.pushBack(key);
        table.addOrReplace(key,value);
    }

    proc createWith(in key: keyType, in value: valType): dict(keyType,valType) {
        var newDict = this;
        newDict.insert(key,value);
        return newDict;
    }

    proc ref remove(in key: keyType) {
        table.remove(key);
        order.remove(key);
    }

    proc createWithout(in key: keyType): dict(keyType,valType) {
        var ord: list(keyType);
        var tbl: map(keyType,valType);
        for k in this.keys() {
            if k != key then {
                ord.pushBack(k);
                tbl.addOrReplace(k,this[k]);
            }
        }
        return new dict(tbl,ord);
    }

}
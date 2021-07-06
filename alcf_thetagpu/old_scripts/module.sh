
_module_comgen_words_and_files ()
{
    local k=0;
    local setnospace=1;
    for val in $(compgen -W "$1" -- "$2");
    do
        if [ !$setnospace -a "${val: -1:1}" = '/' ]; then
            type compopt &>/dev/null && compopt -o nospace;
            setnospace=0;
        fi;
        COMPREPLY[k++]="$val";
    done
}

_modules ()
{
    local modpath;
    modpath=/lib/modules/$1;
    COMPREPLY=($( compgen -W "$( command ls -RL $modpath 2>/dev/null |         sed -ne 's/^\(.*\)\.k\{0,1\}o\(\.[gx]z\)\{0,1\}$/\1/p' )" -- "$cur" ))
}

_module_raw ()
{
    unset _mlshdbg;
    if [ "${MODULES_SILENT_SHELL_DEBUG:-0}" = '1' ]; then
        case "$-" in
            *v*x*)
                set +vx;
                _mlshdbg='vx'
            ;;
            *v*)
                set +v;
                _mlshdbg='v'
            ;;
            *x*)
                set +x;
                _mlshdbg='x'
            ;;
            *)
                _mlshdbg=''
            ;;
        esac;
    fi;
    unset _mlre _mlIFS;
    if [ -n "${IFS+x}" ]; then
        _mlIFS=$IFS;
    fi;
    IFS=' ';
    for _mlv in ${MODULES_RUN_QUARANTINE:-};
    do
        if [ "${_mlv}" = "${_mlv##*[!A-Za-z0-9_]}" -a "${_mlv}" = "${_mlv#[0-9]}" ]; then
            if [ -n "`eval 'echo ${'$_mlv'+x}'`" ]; then
                _mlre="${_mlre:-}${_mlv}_modquar='`eval 'echo ${'$_mlv'}'`' ";
            fi;
            _mlrv="MODULES_RUNENV_${_mlv}";
            _mlre="${_mlre:-}${_mlv}='`eval 'echo ${'$_mlrv':-}'`' ";
        fi;
    done;
    if [ -n "${_mlre:-}" ]; then
        eval `eval ${_mlre}/usr/bin/tclsh /usr/share/Modules/libexec/modulecmd.tcl bash '"$@"'`;
    else
        eval `/usr/bin/tclsh /usr/share/Modules/libexec/modulecmd.tcl bash "$@"`;
    fi;
    _mlstatus=$?;
    if [ -n "${_mlIFS+x}" ]; then
        IFS=$_mlIFS;
    else
        unset IFS;
    fi;
    unset _mlre _mlv _mlrv _mlIFS;
    if [ -n "${_mlshdbg:-}" ]; then
        set -$_mlshdbg;
    fi;
    unset _mlshdbg;
    return $_mlstatus
}

module ()
{
    _module_raw "$@" 2>&1
}

_module_avail ()
{
    local cur="${1:-}";
    if [ -z "$cur" -o "${cur:0:1}" != '-' ]; then
        module avail --color=never -s -t -S --no-indepth $cur 2>&1 | sed '
            /^-\+/d; /^\s*$/d;
            /->.*$/d;
            /:$/d;
            s#^\(.*\)/\(.\+\)(.*default.*)#\1\n\1\/\2#;
            s#(.*)$##g;
            s#\s*$##g;';
    fi
}

_module_savelist ()
{
    module savelist --color=never -s -t 2>&1 | sed '
        /No named collection\.$/d;
        /Named collection list$/d;
        /:$/d;'
}

_module_not_yet_loaded ()
{
    _module_avail ${1:-} | sort | sed -E "\%^(${LOADEDMODULES//:/|})$%d"
}

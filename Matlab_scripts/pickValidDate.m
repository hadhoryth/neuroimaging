function o_idx = pickValidDate(year, month, b, selector)
    year = abs(year - b.year);
    pos = find(year(:) < 1);
    o_idx = pos;
    if (length(pos) == 1)
        return;
    end
    month = month - b.month(pos);
    idx = find(month(:) < 0 & abs(month(:)) < 6);
    if(isempty(idx) == 0)
        %             if(length(idx) > 1 && strcmp(selector,'all') == 0)
        %                idx = idx(selector);
        %             end
        o_idx = pos(idx);
    end
end
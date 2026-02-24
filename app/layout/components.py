from dash import html, dcc


def create_card(children, title=None, card_id=None):
    """Create an Apple-style card component"""
    card_content = []
    
    if title:
        card_content.append(html.H3(title, className='card-title'))
    
    if isinstance(children, list):
        card_content.extend(children)
    else:
        card_content.append(children)
    
    props = {'className': 'card'}
    if card_id is not None:
        props['id'] = card_id
    
    return html.Div(card_content, **props)


def create_control_group(label, component, help_text=None):
    """Create a labeled control group"""
    elements = [
        html.Label(label, className='control-label')
    ]
    
    if help_text:
        elements.append(html.Div(help_text, className='help-text'))
    
    elements.append(component)
    
    return html.Div(elements, className='control-group')


def create_alert(message, alert_type='info'):
    """
    Create an alert box
    
    Args:
        message: Alert message
        alert_type: 'info', 'success', 'warning', or 'error'
    """
    return html.Div(
        message,
        className=f'alert alert-{alert_type}'
    )


def create_section_header(title, subtitle=None):
    """Create a section header"""
    elements = [html.H2(title, className='section-title')]
    
    if subtitle:
        elements.append(html.P(subtitle, className='section-subtitle'))
    
    return html.Div(elements, className='section-header')


def create_button(text, button_id, button_type='primary', disabled=False):
    """Create a styled button"""
    return html.Button(
        text,
        id=button_id,
        className=f'btn btn-{button_type}',
        disabled=disabled
    )


def create_slider(slider_id, min_val, max_val, step, value, label=None, marks=None):
    """Create a slider with optional label"""
    # Auto-generate marks if not provided
    if marks is None:
        # Calculate appropriate number of marks based on range
        range_size = max_val - min_val
        
        # Adaptive mark calculation for cleaner display
        if range_size <= 2:
            # Small range: show 3-5 marks
            mark_step = 0.5
        elif range_size <= 10:
            # Medium range: show ~5 marks
            mark_step = 2
        elif range_size <= 100:
            # Large range: show ~5 marks
            mark_step = 25
        elif range_size <= 1000:
            # Very large range (like smooth_iter 0-5000): show 5-6 marks
            mark_step = 1000
        else:
            # Extremely large range: show max 5 marks
            mark_step = range_size / 5
        
        # Generate marks - only show start, end, and a few key points
        marks = {}
        current = min_val
        
        # Add min value
        if abs(min_val) < 10 and step < 1:
            marks[min_val] = f'{min_val:.1f}'
        else:
            marks[min_val] = f'{int(min_val)}'
        
        # Add intermediate marks
        while current < max_val:
            current += mark_step
            if current >= max_val:
                break
            # Format based on value size
            if abs(current) < 10 and step < 1:
                marks[current] = f'{current:.1f}'
            else:
                marks[current] = f'{int(current)}'
        
        # Always add max value
        if abs(max_val) < 10 and step < 1:
            marks[max_val] = f'{max_val:.1f}'
        else:
            marks[max_val] = f'{int(max_val)}'
    
    slider = dcc.Slider(
        id=slider_id,
        min=min_val,
        max=max_val,
        step=step,
        value=value,
        marks=marks,
        tooltip={"placement": "bottom", "always_visible": True}
    )
    
    if label:
        return create_control_group(label, slider)
    
    return slider


def create_dropdown(dropdown_id, options, value=None, placeholder='Select...', label=None, multi=False):
    """Create a dropdown with optional label"""
    dropdown = dcc.Dropdown(
        id=dropdown_id,
        options=options,
        value=value,
        placeholder=placeholder,
        multi=multi,
        clearable=True,
        className='dropdown'
    )
    
    if label:
        return create_control_group(label, dropdown)
    
    return dropdown


def create_radio_items(radio_id, options, value, label=None, inline=True):
    """Create radio buttons with optional label"""
    radio = dcc.RadioItems(
        id=radio_id,
        options=options,
        value=value,
        inline=inline,
        className='radio-items'
    )
    
    if label:
        return create_control_group(label, radio)
    
    return radio


def create_checkbox(checkbox_id, label_text, value=False):
    """Create a checkbox with label"""
    return dcc.Checklist(
        id=checkbox_id,
        options=[{'label': label_text, 'value': 'checked'}],
        value=['checked'] if value else [],
        className='checkbox'
    )


def create_input(input_id, input_type='text', placeholder='', value='', label=None):
    """Create an input field with optional label"""
    input_field = dcc.Input(
        id=input_id,
        type=input_type,
        placeholder=placeholder,
        value=value,
        className='input-field'
    )
    
    if label:
        return create_control_group(label, input_field)
    
    return input_field


def create_loading(component, loading_id='loading'):
    """Wrap component in loading spinner"""
    return dcc.Loading(
        id=loading_id,
        type='circle',
        children=component,
        className='loading-wrapper'
    )
